"""
in this class i want to get a pdf page (from a document)
and a coordinates of step rectangle

from this i should exctrat
1) all text element
2) all image element

go over text, and get:
1) the step number (by font) -- set at the begging
2) amount of parts __x
3) others - TBD

then go to images and get:
1) for each amount of part __x - the image above
2) the bigest image - be it the main part
3) all others - TBD


"""

import os
import fitz  # PyMuPDF
from dataclasses import dataclass
from typing import Tuple, Optional, Literal

import numpy as np
import cv2
from skimage.metrics import structural_similarity as ssim

@dataclass
class ContentBlock:
    pos: Tuple[int, int, int, int]                      # (x1, y1, x2, y2)
    type: Literal["image", "text"]                      # 'image' or 'text'
    image_extension: Optional[str] = None               # e.g., 'png', 'jpg'
    xref: Optional[str] = None
    text: Optional[str] = None                          # extracted or associated text
    font: Optional[int] = None
    image_data: Optional[bytes] = None                  # binary image data
    used: bool = False  # Default to unused

def get_area(cb: ContentBlock) -> int:
    x1, y1, x2, y2 = cb.pos
    return abs((x2 - x1) * (y2 - y1))  # abs to guard against invalid coords


def rect_intersect(step_rect, target):
    step_x0, step_y0, step_x1, step_y1 = step_rect
    tar_x0, tar_y0, tar_x1, tar_y1 = target

    if ((step_x0 <= tar_x0 <= step_x1 or step_x0 <= tar_x1 <= step_x1) and
            (step_y0 <= tar_y0 <= step_y1 or step_y0 <= tar_y1 <= step_y1)):
        return True
    return False

min_acceptance_score = 0.8
class StepProcessor:
    def __init__(self,caller):
        self.submodules_rects = None
        self.parts_df = None

        self.pdf_document = None
        self.pdf_page = None
        self.steps_rects = None
        self.step_font_size = caller.instruction_step_font_size
        self.parts_font_size = caller.parts_font_size
        self.numbered_substep_font_size = caller.numbered_substep_font_size



    def __extract_relevant_images(self,step_rect):
        # --- Images ---
        step_images = []
        # print("\nImages:")
        images = self.pdf_page.get_images(full=True)
        if images:
            for i, img in enumerate(images):
                xref = img[0]
                #
                rect = self.pdf_page.get_image_bbox(img)

                if rect_intersect(step_rect, rect):
                    base_image = self.pdf_document.extract_image(xref)

                    image_block = ContentBlock(
                        pos=(int(rect.x0), int(rect.y0), int(rect.x1), int(rect.y1)),
                        type="image",
                        xref=xref,
                        image_data=base_image["image"],
                        image_extension=base_image["ext"]
                    )
                    step_images.append(image_block)

        return step_images

    def __extract_relevant_text(self,step_rect):

        # print("\nText:")
        step_text = []
        text_dict = self.pdf_page.get_text("dict")
        for block in text_dict["blocks"]:
            for line in block.get("lines", []):
                for span in line["spans"]:
                    rect = span["bbox"]

                    if rect_intersect(step_rect, rect):
                        text_block = ContentBlock(
                            pos = [int(v) for v in rect],
                            type="text",
                            text= span["text"].strip(),
                            font = span['size']
                        )
                        step_text.append(text_block)
                        # print ("text found: ",text_block)

        return step_text

    def __compare_images_ssim(self,img1, img2):
        # Resize img2 to match img1
        img2_resized = cv2.resize(img2, (img1.shape[1], img1.shape[0]))

        # Convert to grayscale
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(img2_resized, cv2.COLOR_BGR2GRAY)

        # Calculate SSIM
        score, _ = ssim(gray1, gray2, full=True)
        return score

    def extract_step(self,top_folder,pdf_document, page_number, steps_rects,parts_df,submodules_rects):
        self.pdf_document = pdf_document
        self.pdf_page = pdf_document[page_number]
        self.steps_rects = steps_rects
        self.parts_df = parts_df
        self.submodules_rects = submodules_rects

        # print ("steps: ",steps_rects)
        # print ("submodules: ",submodules_rects)

        parts_file = None

        for step in self.steps_rects:
            texts = self.__extract_relevant_text(step)
            images = self.__extract_relevant_images(step)

            #get relevant submodules (they are rects, not objects)
            submodules = []
            for each_submodule in submodules_rects:
                if rect_intersect(step, each_submodule):
                    submodules.append(each_submodule)

            print ("submodules: ",submodules)
            '''
            in each step may be more than one submodule. and they may not have numbers
            actualy, they will not have numbers if they are rounded submodules
            '''



            # nado proverit' ot kakogo shaga etot sub modul.
            # potomu chto na stranitze est' 2 shaga. no submodule tol'ko v odnom iz nih
            #
            # kstati. na stranitze mogut bit 2 sabmodula ot dvuh raznih shagov. a mogut bit i ot odnogo
            #
            # kak bi tak sdelat' chtobi submodule obrabativalsya takje kak i prosto modul

            # if self.submodules_rects:
            #     submodule_images = self.__extract_relevant_submodule_images(submodules_rects[0])
            #     print ("submodule_images",submodule_images)

            # !!!!!!!!!!!1__extract_relevant_sub_module

            step_number = 0
            image_counter = 0

            # !!! somehow, we should address sub,and sub-sub steps here
            # and do remove used text and images, so they don't interfer'

            # get step number, used for the folder name
            for text in texts:
                if text.font == self.step_font_size:
                    step_number = text.text
                    folder = 's' + str(step_number)
                    folder = os.path.join(top_folder, folder)
                    os.makedirs(folder, exist_ok=True)

                    # create a file for used parts list
                    parts_file_path = os.path.join(folder, 'step_parts.csv')
                    parts_file = open(parts_file_path, 'w')
                    parts_file.write("element_id,amount,part_num,color_id\n")
                    break

            print(f"step number is: {step_number} , texts: {len(texts)} , images: {len(images)}")


            # get amount of parts,and pictures
            for text in texts:
                if text.font == self.parts_font_size and 'x' in text.text:
                    # here we should get the images of required parts
                    tx0, ty0, tx1, ty1 = text.pos
                    image_counter += 1

                    for each in images:
                        x0, y0, x1, y1 = each.pos
                        if x0 < tx0 + 5 < x1 and y0 < ty0 - 5 < y1:
                            image_name = f"prt_{image_counter}_{text.text}.{each.image_extension}"
                            image_path = os.path.join(folder, image_name)
                            each.used = True
                            with open(image_path, "wb") as img_file:
                                img_file.write(each.image_data)

                            #here we want to find each step element_id, and part_id
                            nparr_step = np.frombuffer(each.image_data, np.uint8)
                            img_step = cv2.imdecode(nparr_step, cv2.IMREAD_COLOR)

                            max_score = -2
                            max_matching_part = None

                            for index, row in self.parts_df.iterrows():
                                img_part = row['image']
                                if img_part is not None:
                                    score = self.__compare_images_ssim(img_part, img_step)
                                    if score > max_score:
                                        max_score = score
                                        max_matching_part = row

                            if max_matching_part is not None:
                                # print(f"Part {max_matching_part['element_id']}, score {max_score}")

                                if max_score > min_acceptance_score:
                                    amount = text.text.replace('x','')
                                    parts_file.write(f"{max_matching_part['element_id']},{amount},"
                                                     f"{max_matching_part['part_num']},{max_matching_part['color_id']},\n")
                                else:
                                    print ("part match failed with score ",max_score)

                                # cv2.imshow('img_part', max_matching_part['image'])
                                # cv2.imshow('img_step', img_step)
                                # cv2.waitKey(0)
                                # cv2.destroyAllWindows()


                            continue

            parts_file.close()

            # get the biggest image. it's probably the step image
            sorted_images = sorted(images, key=get_area, reverse=True)

            image_name = f"big_step.{sorted_images[0].image_extension}"
            image_path = os.path.join(folder, image_name)
            sorted_images[0].used = True
            with open(image_path, "wb") as img_file:
                img_file.write(sorted_images[0].image_data)





            # here we left with unused images, that are probably from the submodule
            for i,submodule in enumerate(submodules):

                '''
                up to here, we parsed the main step ----------- start submodule
                rounded submodule can be with with numbers, but also can be without.
                and can be with '2x'.

                if there is _x, it probably should be in submodule name
                '''

                submodule_texts = self.__extract_relevant_text(submodule)
                submodule_multiplier = "1x"
                if submodule_texts:
                    # look for _x text
                    for text in submodule_texts:
                        if 'x' in text.text:
                            submodule_multiplier = text.text



                # i want to use the same images array, so we left with true unused images
                submodule_images = []
                for image in sorted_images:
                    image_rec = image.pos
                    if rect_intersect(submodule, image_rec):
                        submodule_images.append(image)


                #create subfolder
                subfolder_path = os.path.join(folder, "sub"+str(i)+"_"+submodule_multiplier)
                os.makedirs(subfolder_path, exist_ok=True)

                #write all images here
                image_counter = 0
                for each in submodule_images:

                    image_name = f"sub_{image_counter}.{each.image_extension}"
                    image_counter += 1
                    image_path = os.path.join(subfolder_path, image_name)
                    each.used = True
                    with open(image_path, "wb") as img_file:
                        img_file.write(each.image_data)

                remarks = []
                for each in submodule_texts:
                    remarks.append(each.text)

                if remarks:
                    text = ', '.join(remarks)
                    image_path = os.path.join(subfolder_path, "remarks.txt")
                    with open(image_path, "w") as txt_file:
                        txt_file.write(text)


            # get all unused images
            image_counter = 0
            for each in sorted_images:
                if each.used == False:
                    image_name = f"imgUU_{image_counter}.{each.image_extension}"
                    image_path = os.path.join(folder, image_name)
                    each.used = True
                    image_counter += 1

                    with open(image_path, "wb") as img_file:
                        img_file.write(each.image_data)





