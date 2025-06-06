'''

the idea of this class is like this:
it's going to take a page from lego instructions, and get some data
for now i know two types:
1) the text embeded in pdf files
2) rounded squares for building parts, and submodules

so actually this is a place for some logic and computer vision
'''
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




        print ("PageProcessor v2 init called")

        self.debug_level = debug_level

        self.step_font_size = None
        self.framed_sub_step_font_size = None
        self.numbered_sub_step_font_size = None
        self.parts_list_font_size = None
        self.page_number_font_size = None

        self.framed_sub_step_color = None
        # self.sub_sub_step_color = None
        self.parts_list_color = None

        # self.page_size = None

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
        self.numbered_sub_step_font_size = meta_dict.get('page_sub_step_font_size', 22)
        self.framed_sub_step_font_size = meta_dict.get('sub_step_font_size', 16)
        self.parts_list_font_size = meta_dict.get('parts_list_font_size',8)
        self.page_number_font_size = meta_dict.get('page_number_font_size',10)

        self.framed_sub_step_color = meta_dict.get('sub_step_color', [202, 239, 255])
        # self.sub_sub_step_color = meta_dict.get('step_font_size',None)
        self.parts_list_color = meta_dict.get('parts_list_color',[242,215,182])
        # self.parts_list_color = meta_dict.get('parts_list_color', [255,255,255])

        self.rotate_icon_img = cv2.imread('rotate_v2.png')

    def prepare_page(self,pdf_doc,page_index):
        '''
        in this function, we ONLY PREPARE data:
        so we have page,and meta data.
        our goal is:
        1) extract all page texts
        2) extract all page images
        3) extract all frames

        4) store them in a processable object
        '''


        ##################################

        page = pdf_doc[page_index]

        # self.page_size = (int(page.rect.width), int(page.rect.height))

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
                print (f"image {xref} dropped")
                continue

            # area = abs((rect.x1 - rect.x0) * (rect.y1 - rect.y0))
            # print(f"xref: {img[0]} with area {area}")

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

        #get substep frames
        self.framed_sub_steps_list = self.__get_frames_locations(page_image, self.framed_sub_step_color)

        #get parts lists, and other
        self.other_frames_list = self.__get_frames_locations(page_image, self.parts_list_color)

        # get rotation icon
        self.rotate_icons_list = self.__get_rotate_icon_location(page_image,self.rotate_icon_img)

    def is_image_empty(self,image_bytes, threshold=5):
        # Convert bytes to NumPy array
        nparr = np.frombuffer(image_bytes, np.uint8)

        # Decode image from buffer (automatically handles PNG/JPEG/etc.)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Invalid image bytes or unsupported format.")

        # Flatten image to list of pixels: shape (num_pixels, 3)
        pixels = img.reshape(-1, img.shape[-1])

        # Count unique colors
        unique_colors = np.unique(pixels, axis=0)

        # Return True if the number of unique colors is below the threshold
        return len(unique_colors) < threshold

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
        self.step_texts.clear()
        self.frame_sub_module_texts.clear()
        self.numbered_sub_module_texts.clear()
        self.parts_list_texts.clear()
        self.other_texts.clear()

        self.images_list.clear()
        self.other_frames_list.clear()
        self.parts_list.clear()
        self.framed_sub_steps_list.clear()
        self.rotate_icons_list.clear()

        self.steps.clear()


        self.prepare_page(pdf_doc,page_index)

        self.detect_parts_list()

        # self.steps = self.detect_steps_area()

        if self.debug_level > 0:
            self.__show_debug_frames(self.page_image)


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


    '''
    here we get stages number list, in a form of (text,position)
    and a partslist list (position)
    
    we need to applay some logic,to extend stages number are to the whole stage are
    and return a new list
    '''

    # def detect_steps_area(self):
    #
    #     # tut nado peredelat tak:
    #     # 1) tepre parts list i prochie list - eto ob'ekti'
    #     # 2) tut shag 1 - proverit esli est' nomera fontov ot shaga
    #     # 3) tut nado proverit' esli est' shag ot sub module s nomerom.
    #     #
    #     # on mojet bit kak i samostoyatelniy, tak i posle bolshogo shaga
    #     #
    #     # a escho ya sovsem ne produmal kak nahodit font vnutri rozovogo submodule. mojet tam drugoy ?
    #     # ili takoyje ? voobshem obdumat
    #
    #     pdf_x, pdf_y = self.page_size
    #     steps = []
    #
    #     # exclude some obvious cases
    #
    #     # if there is no steps numbers - return empty array
    #
    #     if len(self.step_texts) == 0 and len(self.page_sub_module_texts) == 0: return steps
    #
    #     if len(self.step_texts) > 0:
    #         # means we have some real steps,
    #         # and we can check if there is parts list
    #
    #         # sort the list, so the last step on the page, will be last on array
    #         self.step_texts.sort(key=lambda block: int(block.text))
    #
    #         # step 1 - detect the top left point - start of step
    #
    #         for step_number in self.step_texts:
    #             x1, y1, x2, y2 = step_number.pos
    #
    #             # txt, (x1, y1, x2, y2)
    #             # check if there is a part list above it
    #             for part_list in self.parts_list:
    #                 rx1, ry1, rx2, ry2 = part_list.pos
    #                 if rx1 < x1 + 10 < rx2 and ry1 < y1 - 20 < ry2:
    #                     y1 = ry1 - 1
    #                     x2 = rx2
    #                     break
    #
    #             # make a few pixels before the number inside step area
    #             x1 = x1 - 10
    #             steps.append((x1, y1, x2, y2))
    #
    #         for i in range(len(steps) - 1):
    #             x1, y1, x2, y2 = steps[i]
    #
    #             index_right = -1
    #             index_down = -1
    #
    #             # first check if THE NEXT ONE is below
    #             xx1, yy1, xx2, yy2 = steps[i + 1]
    #
    #             if y2 < yy1:
    #                 # print(" found down of NEXT")
    #                 index_down = i + 1
    #
    #             for j in range(i + 1, len(steps)):
    #                 xx1, yy1, xx2, yy2 = steps[j]
    #
    #                 if x2 < xx1 and index_right == -1:
    #                     # print(" found right neighbor")
    #                     index_right = j
    #
    #             # update from relevant neighbor
    #             if index_right != -1:
    #                 xx1, yy1, xx2, yy2 = steps[index_right]
    #                 x2 = xx1 - 5
    #             else:
    #                 x2 = pdf_x - 5
    #
    #             if index_down != -1:
    #                 xx1, yy1, xx2, yy2 = steps[index_down]
    #                 y2 = yy1 - 5
    #             else:
    #                 y2 = pdf_y - 5
    #
    #             steps[i] = (x1, y1, x2, y2)
    #
    #         # update the last element size
    #
    #         x1, y1, x2, y2 = steps[-1]
    #         steps[-1] = (x1, y1, pdf_x - 5, pdf_y - 5)
    #
    #     return steps
