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

    # If one rectangle is to the left of the other
    if ((step_x0 <= tar_x0 <= step_x1 or step_x0 <= tar_x1 <= step_x1) and
            (step_y0 <= tar_y0 <= step_y1 or step_y0 <= tar_y1 <= step_y1)):
        return True
    return False


class StepProcessor:
    def __init__(self,step_font_size, parts_font_size):
        self.pdf_document = None
        self.pdf_page = None
        self.steps_rects = None
        self.step_font_size = step_font_size
        self.parts_font_size = parts_font_size

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

    def extract_step(self,pdf_document, page_number, steps_rects):
        self.pdf_document = pdf_document
        self.pdf_page = pdf_document[page_number]
        self.steps_rects = steps_rects


        for step in steps_rects:
            texts = self.__extract_relevant_text(step)
            images = self.__extract_relevant_images(step)



            step_number = 0
            image_counter = 0

            # get step number, used for the folder name
            for text in texts:
                if text.font == self.step_font_size:
                    step_number = text.text
                    folder = 's' + str(step_number)
                    os.makedirs(folder, exist_ok=True)
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

                            # print("parts amount: ", text.text[:-1])

                else:
                    print ("unknown text: ",text.text)

            # get the biggest image. it's probably the step image
            sorted_images = sorted(images, key=get_area, reverse=True)

            image_name = f"big_step.{sorted_images[0].image_extension}"
            image_path = os.path.join(folder, image_name)
            sorted_images[0].used = True
            with open(image_path, "wb") as img_file:
                img_file.write(sorted_images[0].image_data)

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

        print ("export done")




