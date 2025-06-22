'''

the idea of this class is like this:
it's going to take a page from lego instructions, and get some data
for now i know two types:
1) the text embeded in pdf files
2) rounded squares for building parts, and submodules

so actually this is a place for some logic and computer vision
'''
import time

import fitz
import numpy as np
import cv2
import warnings
import os

from dataclasses import dataclass
from typing import Tuple, Optional, Literal

from RotateIconMathcher import IconFinder

'''
this is a simple data holder class. it a common clas for texts,frames,and images
'''
@dataclass
class ContentBlock:
    type: Literal["image", "frame", "text"]

    #frame (and others)
    pos: Tuple[int, int, int, int]

    #text elements
    text: Optional[str] = None  # extracted or associated text
    font_size: Optional[int] = None

    #image elements
    image_extension: Optional[str] = None               # e.g., 'png', 'jpg'
    xref: Optional[str] = None
    image_bytes: Optional[bytes] = None  # binary image data
    image_name: Optional[str] = None

    used: bool = False  # Default to unused

    def __str__(self):
        return (
            f"ContentBlock(type={self.type}, pos={self.pos}, text={self.text}, "
            f"font_size={self.font_size}, image_extension={self.image_extension}, "
            f"xref={self.xref}, used={self.used})"
        )

    __repr__ = __str__


def is_inside_rect(px,py, rect):
    rx1,ry1,rx2,ry2 = rect
    return rx1 < px < rx2 and ry1 < py < ry2

def rect_intersect(rect1, rect2):
    step_x0, step_y0, step_x1, step_y1 = rect1
    tar_x0, tar_y0, tar_x1, tar_y1 = rect2

    if ((step_x0 <= tar_x0 <= step_x1 or step_x0 <= tar_x1 <= step_x1) and
            (step_y0 <= tar_y0 <= step_y1 or step_y0 <= tar_y1 <= step_y1)):
        return True
    return False

'''
here is the basic pdf PAGE processing:
on init, we need to know meta data
1) step font size
2) sub step font size
3) parts list font size

4) parts list color
5) sub module color
6) sub sub module color (in general, as a parts list)

7) special marks image - like the turn around/upside down

8) page number font size [optional]


we want to get 
1) pdf doc
2) page number

we want to return:
1) list of texts
2) list of images
3) list of frames
4) list of parts_lists ??

5) debug info: clean page, and full details page

'''


class PageProcessor:
    def __init__(self, debug_level = 0):

        self.debug_level = debug_level

        self.step_font_size = None
        self.framed_sub_step_font_size = None
        self.numbered_sub_step_font_size = None
        self.parts_list_font_size = None
        self.page_number_font_size = None

        self.framed_sub_step_color = None
        # self.sub_sub_step_color = None
        self.parts_list_color = None

        self.page_size = None

        # self.texts_list = []
        # instead of all common texts, there is few lists: step number, sub module,page_sub_moduele,others
        # f.ex if we have 1:1 in other frames - drop it
        self.step_texts = []
        self.frame_sub_module_texts = []
        self.numbered_sub_module_texts = []
        self.parts_list_texts = []
        self.other_texts = []

        self.images_list = []
        self.other_frames_list = []
        self.parts_list = []
        self.framed_sub_steps_list = []
        self.rotate_icons_list = []

        self.steps = []

        self.page_image = None

        self.rotate_icon_img = None

    def set_meta(self,meta_dict):

        self.step_font_size = meta_dict.get('step_font_size',26)
        self.numbered_sub_step_font_size = meta_dict.get('numbered_sub_step_font_size', 22)
        self.framed_sub_step_font_size = meta_dict.get('sub_step_font_size', 16)
        self.parts_list_font_size = meta_dict.get('parts_list_font_size',8)
        self.page_number_font_size = meta_dict.get('page_number_font_size',10)

        self.framed_sub_step_color = meta_dict.get('sub_step_color', [202, 239, 255])
        # self.sub_sub_step_color = meta_dict.get('step_font_size',None)
        self.parts_list_color = meta_dict.get('parts_list_color',[242,215,182])
        # self.parts_list_color = meta_dict.get('parts_list_color', [255,255,255])

        self.rotate_icon_img = cv2.imread('rotate_v2.png')

    def prepare_page(self,pdf_doc,page_index):

        page = pdf_doc[page_index]

        self.page_size = (int(page.rect.width), int(page.rect.height))

        # if self.debug_level > 0:
        #     folder = f"debug/page_{page_index + 1}"
        #     os.makedirs(folder, exist_ok=True)

        # extract all texts (from pdf structure)
        for block in page.get_text("dict")["blocks"]:
            if "lines" in block:
                for line in block["lines"]:
                    for span in line["spans"]:
                        text_block = ContentBlock(
                            type="text",
                            pos=[int(v) for v in span["bbox"]],
                            text=span["text"].strip(),
                            font_size=span['size']
                        )
                        # self.texts_list.append(text_block)
                        if span['size'] == self.step_font_size:
                            self.step_texts.append(text_block)
                        elif span['size'] == self.framed_sub_step_font_size:
                            self.frame_sub_module_texts.append(text_block)
                        elif span['size'] == self.numbered_sub_step_font_size:
                            self.numbered_sub_module_texts.append(text_block)
                        elif span['size'] == self.parts_list_font_size:
                            self.parts_list_texts.append(text_block)
                        else:
                            self.other_texts.append(text_block)

        # some gpt code, to remove duplicates from page_sub_module_texts
        seen = set()
        result = []
        for obj in self.numbered_sub_module_texts:
            key = (obj.text, tuple(obj.pos) if isinstance(obj.pos, list) else obj.pos)
            if key not in seen:
                seen.add(key)
                result.append(obj)
        self.numbered_sub_module_texts = result

        #and for frames sub steps
        seen = set()
        result = []
        for obj in self.frame_sub_module_texts:
            key = (obj.text, tuple(obj.pos) if isinstance(obj.pos, list) else obj.pos)
            if key not in seen:
                seen.add(key)
                result.append(obj)
        self.frame_sub_module_texts = result

        # extract all images (from pdf structure)
        images = page.get_images(full=True)

        for img_index, img in enumerate(images, start=1):
            xref = img[0]
            base_image = pdf_doc.extract_image(xref)
            rect = page.get_image_bbox(img)
            bbox = (int(rect.x0), int(rect.y0), int(rect.x1), int(rect.y1))


            if self.is_image_empty(base_image["image"]):
                # print (f"image {xref} dropped by colors")
                continue

            # if self.is_almost_empty(base_image["image"]):
            #     print(f"image {xref} dropped by pixels")
            #     continue

            # area = abs((rect.x1 - rect.x0) * (rect.y1 - rect.y0))
            # if area < 250:
            #     print(f"xref: {img[0]} with area {area}")
            #     continue

            image_block = ContentBlock(
                type="image",
                pos=bbox,

                image_extension=base_image["ext"],
                xref = img[0],
                image_bytes=base_image["image"]
            )
            self.images_list.append(image_block)

        # find and extract frames (rounded boxes) - by openCV

        # get image of the page
        pix = page.get_pixmap(matrix=fitz.Matrix(1, 1))
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3)
        # Create a writable copy to avoid read-only error
        page_image = img.copy()
        page_image = cv2.cvtColor(page_image, cv2.COLOR_BGRA2RGB)
        self.page_image = page_image

        self.images_list = self.remove_overlapping_images(self.images_list,debug = False)

        # match images
        matching_ids = self.find_possible_matches(self.images_list)
        merged = self.merge_matched_images_cv(matching_ids,self.images_list)

        self.images_list = merged
        # for xref_a, xref_b, img_np in merged:
        #     cv2.imshow(f"Merged {xref_a} + {xref_b}", img_np)
        #     cv2.waitKey(0)

        #
        #
        # zz = find_possible_matches(self.images_list)
        # print (zz)

        #get substep frames
        self.framed_sub_steps_list = self.__get_frames_locations(page_image, self.framed_sub_step_color)

        #get parts lists, and other
        self.other_frames_list = self.__get_frames_locations(page_image, self.parts_list_color)

        # get rotation icon
        self.rotate_icons_list = self.__get_rotate_icon_location(page_image,self.rotate_icon_img)

    def is_image_empty(self,image_bytes, threshold=35):
        # Convert bytes to NumPy array and decode image
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Invalid image bytes or unsupported format.")

        # Flatten pixels
        pixels = img.reshape(-1, img.shape[-1])

        seen = set()
        for pixel in pixels:
            seen.add(tuple(pixel))
            if len(seen) > threshold:
                return False  # early exit, image is not empty

        return True  # didn't reach threshold

    def is_almost_empty(self, image_bytes, threshold_percent=15, pixel_threshold=10):
        # Convert bytes to NumPy array
        nparr = np.frombuffer(image_bytes, np.uint8)

        # Decode image from buffer (handles PNG/JPEG/etc.)
        img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError("Invalid image bytes or unsupported format.")

        # Normalize pixel values to range [0, 1]
        normalized = img / 255.0

        # Consider pixels non-empty if not near black or white
        non_empty_pixels = np.logical_and(
            normalized > pixel_threshold / 255.0,
            normalized < 1 - (pixel_threshold / 255.0)
        )

        non_empty_count = np.count_nonzero(non_empty_pixels)
        total_pixels = img.size
        non_empty_percent = (non_empty_count / total_pixels) * 100

        return non_empty_percent < threshold_percent

    ######################################

    def process_page(self,pdf_doc,page_index):
        '''
         here comes the logic:
         1) prepare page. this will extract basic elements
         2) do some logic:
            2.1) get parts list and "other" lists
            2.2) with parts list, calculate step area
            2.3) check for numbered_submodule (maybe this means another prepare_extraction)
            2.4) for each step, check for submodule
            2.5) for each submodule, check "other" frames for sub sub module
        '''

        # first , clean data
        # self.step_texts.clear()
        # self.frame_sub_module_texts.clear()
        # self.numbered_sub_module_texts.clear()
        # self.parts_list_texts.clear()
        # self.other_texts.clear()
        #
        # self.images_list.clear()
        # self.other_frames_list.clear()
        # self.parts_list.clear()
        # self.framed_sub_steps_list.clear()
        # self.rotate_icons_list.clear()
        # self.steps.clear()
        #
        # self.prepare_page(pdf_doc,page_index)

        self.detect_parts_list()

        # self.steps = self.detect_steps_area()

        if self.debug_level > 0:
            self.__show_debug_frames(self.page_image)

    def get_image(self,x,y):

        def get_area(block):
            x1, y1, x2, y2 = block.pos
            return (x2 - x1) * (y2 - y1)

        sorted_images = sorted(self.images_list, key=get_area)

        for image in sorted_images:
            out_x0, out_y0, out_x1, out_y1 = image.pos
            if out_x0 <= x <= out_x1 and out_y0 <= y <= out_y1:
                return image
        return None

    def delete_image(self,image):
        self.images_list.remove(image)

    def __show_debug_frames(self,page_image):
        '''
        draw all findings on a pages (maybe few pages)
        we need to draw (and maybe save) several debug images
        1) a clean image of the page
        2) image with all texts detected
        3) image with all images detected (frame + xref)
        4) image with all frames detected (sub steps, and parts colors)

        '''

        # start with the plain image
        # cv2.imshow("clean_page", page_image)

        if self.debug_level >= 1:

            ###### we should improve here, to show steps and subsubmodule, and other stuff

            working_page_basic = page_image.copy()

            # draw sub step frames
            for frame in self.framed_sub_steps_list:
                x0, y0, x1, y1 = frame.pos
                cv2.rectangle(working_page_basic, (x0, y0), (x1, y1), (255, 0, 255), 2)

            # draw parts frames
            for frame in self.parts_list:
                x0, y0, x1, y1 = frame.pos
                cv2.rectangle(working_page_basic, (x0, y0), (x1, y1), (0, 255, 0), 2)

            # draw other frames
            for frame in self.other_frames_list:
                x0, y0, x1, y1 = frame.pos
                cv2.rectangle(working_page_basic, (x0, y0), (x1, y1), (255, 0, 0), 2)

            # draw rotate icon
            for rotate_icon in self.rotate_icons_list:
                x0, y0, x1, y1 = rotate_icon
                cv2.rectangle(working_page_basic, (x0, y0), (x1, y1), (0, 0, 255), 2)

            # # draw steps
            # for step in self.steps:
            #     x0, y0, x1, y1 = step
            #     cv2.rectangle(working_page_basic, (x0, y0), (x1, y1), (128, 128, 0), 2)

            cv2.imshow("basic_debug_page_processor", working_page_basic)


        if self.debug_level >=2 : # 2+

            working_page_text_images = page_image.copy()

            # draw all texts

            for text in self.step_texts:
                x0, y0, x1, y1 = text.pos
                cv2.rectangle(working_page_text_images, (x0, y0), (x1, y1), (0, 255, 0), 2)
            for text in self.frame_sub_module_texts:
                x0, y0, x1, y1 = text.pos
                cv2.rectangle(working_page_text_images, (x0, y0), (x1, y1), (0, 255, 0), 2)
            for text in self.numbered_sub_module_texts:
                x0, y0, x1, y1 = text.pos
                cv2.rectangle(working_page_text_images, (x0, y0), (x1, y1), (0, 255, 0), 2)
            for text in self.parts_list_texts:
                x0, y0, x1, y1 = text.pos
                cv2.rectangle(working_page_text_images, (x0, y0), (x1, y1), (0, 255, 0), 2)
            for text in self.other_texts:
                x0, y0, x1, y1 = text.pos
                cv2.rectangle(working_page_text_images, (x0, y0), (x1, y1), (255, 255, 0), 2)



            # draw all images, with xref

            for image in self.images_list:
                x0, y0, x1, y1 = image.pos
                cv2.rectangle(working_page_text_images, (x0, y0), (x1, y1), (0, 0, 255), 2)
                cv2.putText(working_page_text_images, str(image.xref) + '.', (x0, y1 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (0, 0, 255), 2)
            cv2.imshow("page_with_text_&_images", working_page_text_images)



        cv2.waitKey(1)


    def detect_parts_list(self):
        '''
          the logic here is so:
          get a list of parts candidate = frames
          once you have a frame in hand, look for text that:
          a) is inside it b) has the '_x' pattern
          you can stop when all frames are done
        '''

        if self.parts_list_color is None:
            warnings.warn("parts list color must be set")
            return None

        verified_parts_list = []

        # reset frame usage
        for frame in self.other_frames_list:
            frame.used = False

        for frame in self.other_frames_list:

            for text in self.parts_list_texts:

                if 'x' in text.text and rect_intersect(frame.pos,text.pos):
                    frame.used = True
                    verified_parts_list.append(frame)
                    #skip to next frame
                    break

        self.parts_list = verified_parts_list

        #clean other list
        tmp_list = []
        for frame in self.other_frames_list:
            if not frame.used:
                tmp_list.append(frame)

        self.other_frames_list = tmp_list
        return None

    def __get_rotate_icon_location(self,cv_image,icon_img):

        # Load images
        # icon_img = cv2.imread('rotate_1.png')

        # Create detector
        finder = IconFinder(cv_image.copy(), icon_img, threshold=0.7, scales=[2.0,1.5,1.0, 0.9, 0.8,0.5])

        finder.find_all_matches()  # Collect raw hits
        finder.filter_matches(0.3)  # Run NMS

        return finder.get_filtered_boxes()

    def __get_frames_locations(self,cv_image, color, tolerance = 10):

        # Step 1: calc color bounding
        color_int = np.array(color).astype(int)
        lower = np.clip(color_int - tolerance, 0, 255).astype(np.uint8)
        upper = np.clip(color_int + tolerance, 0, 255).astype(np.uint8)

        # Step 2: Create mask
        mask = cv2.inRange(cv_image, lower, upper)

        # Step 3: Contour detection
        # Convert mask to binary image
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Step 4: Filter by size
        min_area = 1000  # Minimum area to keep
        rects = []
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            area = w * h
            if area > min_area:
                # rects.append((x,y,x+w,y+h))
                frame_block = ContentBlock(
                    type="frame",
                    pos=(x,y,x+w,y+h),
                )
                rects.append(frame_block)

        # Draw rectangles on a copy of the original image
        if self.debug_level >= 3:
            # mask applied
            masked_image = cv_image.copy()
            result = cv2.bitwise_and(masked_image, masked_image, mask=mask)

            # detected rectangles
            rects_image = cv_image.copy()
            for rect in rects:
                x1, y1, x2, y2 = rect.pos
                cv2.rectangle(rects_image, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # detected contours
            contour_image = cv_image.copy()
            cv2.drawContours(contour_image, contours, -1, (0, 0, 255), 2)

            cv2.imshow("Mask", mask)
            cv2.imshow("Result (Masked)", masked_image)
            cv2.imshow("Contours", contour_image)
            cv2.imshow("rect_image", rects_image)

            cv2.waitKey(0)

        return rects



    # a helper function (by gpt) that find two matching images by their location and size
    def find_possible_matches(self,blocks, align_tolerance=1, size_tolerance=2): # 3,5
        """
        Find image blocks that possibly match either horizontally or vertically,
        with both positional and size similarity checks.

        Parameters:
            blocks (list of ContentBlock): Must have 'pos' and 'xref'
            align_tolerance (int): Max pixel gap for alignment
            size_tolerance (int): Max pixel difference for size matching

        Returns:
            list of tuples: (xref_a, xref_b, direction)
        """
        matches = []

        for i, block_a in enumerate(blocks):
            if block_a.type != "image":
                continue
            x1_a, y1_a, x2_a, y2_a = block_a.pos
            width_a = x2_a - x1_a
            height_a = y2_a - y1_a

            for j, block_b in enumerate(blocks):
                if i == j or block_b.type != "image":
                    continue
                x1_b, y1_b, x2_b, y2_b = block_b.pos
                width_b = x2_b - x1_b
                height_b = y2_b - y1_b

                # --- Vertical Match: A above B ---
                vertically_aligned = abs(y2_a - y1_b) <= align_tolerance
                horizontal_overlap = not (x2_a < x1_b or x2_b < x1_a)
                width_similar = abs(width_a - width_b) <= size_tolerance

                if vertically_aligned and horizontal_overlap and width_similar:
                    matches.append((block_a.xref, block_b.xref, "vertical"))

                # --- Horizontal Match: A to the left of B ---
                horizontally_aligned = abs(x2_a - x1_b) <= align_tolerance
                vertical_overlap = not (y2_a < y1_b or y2_b < y1_a)
                height_similar = abs(height_a - height_b) <= size_tolerance

                if horizontally_aligned and vertical_overlap and height_similar:
                    matches.append((block_a.xref, block_b.xref, "horizontal"))

        return matches

    def merge_matched_images_cv(self, matches, blocks, show_preview=True):
        """
        Merge matched image pairs using OpenCV, rescaling to match dimensions.
        Removes original ContentBlocks and adds new merged ones with combined coordinates.

        Parameters:
            matches (list): Tuples of (xref_a, xref_b, direction)
            blocks (list): ContentBlock objects with .xref and .image_bytes
            show_preview (bool): Whether to show preview of merged images

        Returns:
            list: Updated blocks list with merged ContentBlocks
        """
        xref_map = {block.xref: block for block in blocks if block.type == "image"}
        merged_blocks = []
        processed_xrefs = set()
        skip_previews = False  # Flag to skip remaining previews if user presses 'q'

        for xref_a, xref_b, direction in matches:
            block_a = xref_map.get(xref_a)
            block_b = xref_map.get(xref_b)
            if not block_a or not block_b:
                continue

            # Skip if already processed
            if xref_a in processed_xrefs or xref_b in processed_xrefs:
                continue

            # Decode image bytes to OpenCV format
            img_a = cv2.imdecode(np.frombuffer(block_a.image_bytes, np.uint8), cv2.IMREAD_UNCHANGED)
            img_b = cv2.imdecode(np.frombuffer(block_b.image_bytes, np.uint8), cv2.IMREAD_UNCHANGED)

            if img_a is None or img_b is None:
                continue

            # Ensure 3 or 4 channels match
            if img_a.shape[2] != img_b.shape[2]:
                continue

            # --- Rescale to match dimensions before merge ---
            if direction == "vertical":
                # Match width
                target_width = img_a.shape[1]
                scale_b = target_width / img_b.shape[1]
                new_height_b = int(img_b.shape[0] * scale_b)
                img_b_resized = cv2.resize(img_b, (target_width, new_height_b))
                merged = np.vstack((img_a, img_b_resized))

                # Calculate merged coordinates (vertical stacking)
                x1 = min(block_a.pos[0], block_b.pos[0])
                y1 = min(block_a.pos[1], block_b.pos[1])
                x2 = max(block_a.pos[2], block_b.pos[2])
                y2 = max(block_a.pos[3], block_b.pos[3])

            elif direction == "horizontal":
                # Match height
                target_height = img_a.shape[0]
                scale_b = target_height / img_b.shape[0]
                new_width_b = int(img_b.shape[1] * scale_b)
                img_b_resized = cv2.resize(img_b, (new_width_b, target_height))
                merged = np.hstack((img_a, img_b_resized))

                # Calculate merged coordinates (horizontal stacking)
                x1 = min(block_a.pos[0], block_b.pos[0])
                y1 = min(block_a.pos[1], block_b.pos[1])
                x2 = max(block_a.pos[2], block_b.pos[2])
                y2 = max(block_a.pos[3], block_b.pos[3])

            else:
                continue

            # Show preview of merged image
            # if show_preview and not skip_previews:
            #     cv2.imshow(f'Merged Image: {xref_a} + {xref_b}', merged)
            #     print(f"Preview: {xref_a} + {xref_b} ({direction})")


            # Convert merged image back to bytes
            is_success, buffer = cv2.imencode('.png', merged)
            if not is_success:
                continue
            merged_bytes = buffer.tobytes()

            # Create new merged ContentBlock
            merged_block = ContentBlock(
                type="image",
                pos=(x1, y1, x2, y2),
                xref=f"{xref_a}_{xref_b}_merged",  # or generate unique ID
                image_bytes=merged_bytes,
                image_extension="png",
                image_name=f"merged_{block_a.image_name or xref_a}_{block_b.image_name or xref_b}",
                used=False
            )

            merged_blocks.append(merged_block)
            processed_xrefs.add(xref_a)
            processed_xrefs.add(xref_b)

        # Remove original blocks that were merged and add new merged blocks
        updated_blocks = [block for block in blocks if block.xref not in processed_xrefs]
        updated_blocks.extend(merged_blocks)

        return updated_blocks

    # this time by claude. find overlaping images
    import cv2
    import numpy as np

    def remove_overlapping_images(self, blocks, tolerance=5, show_preview=True, max_size_ratio=5.0, debug=True):
        """
        Detect overlapping images, show preview, and remove smaller ones.
        Only removes if the larger image is not more than max_size_ratio times bigger.

        Parameters:
            blocks (list): List of ContentBlock objects
            tolerance (int): Pixel tolerance for considering positions as "same"
            show_preview (bool): Whether to show preview before removing
            max_size_ratio (float): Maximum ratio between larger and smaller image areas (default: 2.0)
            debug (bool): Whether to show debug prints and previews (default: True)

        Returns:
            list: Updated blocks list with overlapping smaller images removed
        """
        image_blocks = [block for block in blocks if block.type == "image"]
        blocks_to_remove = []

        # Override show_preview if debug is off
        if not debug:
            show_preview = False

        def get_area(block):
            """Calculate area of a block"""
            return (block.pos[2] - block.pos[0]) * (block.pos[3] - block.pos[1])

        def positions_overlap(pos1, pos2, tolerance):
            """Check if two positions overlap within tolerance"""
            # Check if starting points are within tolerance
            x1_close = abs(pos1[0] - pos2[0]) <= tolerance
            y1_close = abs(pos1[1] - pos2[1]) <= tolerance

            # Check if one rectangle is contained within another or they significantly overlap
            # Rectangle 1: (x1, y1, x2, y2), Rectangle 2: (x3, y3, x4, y4)
            x1, y1, x2, y2 = pos1
            x3, y3, x4, y4 = pos2

            # Calculate overlap area
            overlap_x = max(0, min(x2, x4) - max(x1, x3))
            overlap_y = max(0, min(y2, y4) - max(y1, y3))
            overlap_area = overlap_x * overlap_y

            # Calculate areas
            area1 = (x2 - x1) * (y2 - y1)
            area2 = (x4 - x3) * (y4 - y3)

            # Consider overlapping if:
            # 1. Starting points are close AND there's significant overlap, OR
            # 2. One rectangle is mostly contained in another (80% overlap)
            significant_overlap = overlap_area > 0.8 * min(area1, area2)

            return (x1_close and y1_close and overlap_area > 0) or significant_overlap

        # Find overlapping pairs
        overlapping_pairs = []
        for i in range(len(image_blocks)):
            for j in range(i + 1, len(image_blocks)):
                block1 = image_blocks[i]
                block2 = image_blocks[j]

                if block1.pos == block2.pos:
                    continue

                if positions_overlap(block1.pos, block2.pos, tolerance):
                    area1 = get_area(block1)
                    area2 = get_area(block2)

                    # Determine which is smaller and calculate size ratio
                    if area1 < area2:
                        smaller_block, larger_block = block1, block2
                        smaller_area, larger_area = area1, area2
                    else:
                        smaller_block, larger_block = block2, block1
                        smaller_area, larger_area = area2, area1

                    # Check if size ratio is within acceptable range
                    size_ratio = larger_area / smaller_area if smaller_area > 0 else float('inf')

                    if size_ratio <= max_size_ratio:
                        overlapping_pairs.append((smaller_block, larger_block, size_ratio))
                    else:
                        if debug:
                            print(
                                f"Skipping overlap removal - size ratio too large: {size_ratio:.2f} > {max_size_ratio}")
                            print(f"  Smaller: {smaller_block.xref} (Area: {smaller_area})")
                            print(f"  Larger:  {larger_block.xref} (Area: {larger_area})")

        if debug:
            print(f"Found {len(overlapping_pairs)} overlapping image pairs")

        # Process each overlapping pair
        for smaller_block, larger_block, size_ratio in overlapping_pairs:
            # Skip if already marked for removal
            if smaller_block in blocks_to_remove:
                continue

            if debug:
                print(f"\nOverlapping images detected (ratio: {size_ratio:.2f}):")
                print(f"  Smaller: {smaller_block.xref} - Area: {get_area(smaller_block)} - Pos: {smaller_block.pos}")
                print(f"  Larger:  {larger_block.xref} - Area: {get_area(larger_block)} - Pos: {larger_block.pos}")

            if show_preview:
                # Decode images for preview
                try:
                    img_small = cv2.imdecode(np.frombuffer(smaller_block.image_bytes, np.uint8), cv2.IMREAD_UNCHANGED)
                    img_large = cv2.imdecode(np.frombuffer(larger_block.image_bytes, np.uint8), cv2.IMREAD_UNCHANGED)

                    if img_small is not None and img_large is not None:
                        # Show both images side by side
                        # Resize for better viewing if needed
                        max_height = 600
                        if img_small.shape[0] > max_height:
                            scale = max_height / img_small.shape[0]
                            new_w = int(img_small.shape[1] * scale)
                            img_small = cv2.resize(img_small, (new_w, max_height))

                        if img_large.shape[0] > max_height:
                            scale = max_height / img_large.shape[0]
                            new_w = int(img_large.shape[1] * scale)
                            img_large = cv2.resize(img_large, (new_w, max_height))

                        # Create side-by-side comparison
                        # Make sure both images have same height for concatenation
                        target_height = max(img_small.shape[0], img_large.shape[0])

                        # Pad smaller height image
                        if img_small.shape[0] < target_height:
                            pad_height = target_height - img_small.shape[0]
                            img_small = np.pad(img_small, ((0, pad_height), (0, 0), (0, 0)), mode='constant',
                                               constant_values=255)

                        if img_large.shape[0] < target_height:
                            pad_height = target_height - img_large.shape[0]
                            img_large = np.pad(img_large, ((0, pad_height), (0, 0), (0, 0)), mode='constant',
                                               constant_values=255)

                        # Add labels
                        if debug:
                            cv2.putText(img_small, f"SMALLER: {smaller_block.xref}", (10, 30),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                            cv2.putText(img_large, f"LARGER: {larger_block.xref}", (10, 30),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                            # Concatenate horizontally
                            comparison = np.hstack([img_small, img_large])

                            cv2.imshow('Overlapping Images - SMALLER (left) will be REMOVED', comparison)

                        blocks_to_remove.append(smaller_block)

                except Exception as e:
                    if debug:
                        print(f"Error showing preview: {e}")
                    blocks_to_remove.append(smaller_block)
            else:
                # Auto-remove without preview
                blocks_to_remove.append(smaller_block)

        # Remove the smaller overlapping images
        updated_blocks = [block for block in blocks if block not in blocks_to_remove]

        if debug:
            print(f"\nRemoved {len(blocks_to_remove)} overlapping smaller images:")
            for block in blocks_to_remove:
                print(f"  - {block.xref} (Area: {get_area(block)})")

        return updated_blocks