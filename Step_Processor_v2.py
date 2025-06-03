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
from typing import Literal, Optional, Tuple, List, Union, Self

import numpy as np
import cv2
from skimage.metrics import structural_similarity as ssim

import Page_Processor_v2

# import Page_Processor_v2

'''
v2 approach. we start our code from step processor:
for now, assume that the first page is the first relevant page, so it contains a step number "1"

we way is so:
init with pdf doc, and page number
take it thrue page_processor. it will internally fill a bunch of lists.
use them to get understanding of steps. note that a step can be on 3-4 pages, 
but a page can include few steps too 

'''
min_acceptance_score = 0.8

@dataclass
class StepBlock:
    type: Literal["step", "page_submodule"]
    step_num: int
    step_frame: Optional[Tuple[int, int, int, int]] = None
    parts_frame: Optional[Tuple[int, int, int, int]] = None
    step_folder_path: Optional[str] = None
    used: bool = False
    # a list of sub steps
    numbered_sub_steps: Optional[List["StepBlock"]] = None  # Self-reference here
    final_image_pos: Optional[Tuple[int, int, int, int]] = None
    final_image_bytes: Optional[bytes] = None  # binary image data




def get_area(cb: Page_Processor_v2.ContentBlock) -> int:
    x1, y1, x2, y2 = cb.pos
    return abs((x2 - x1) * (y2 - y1))  # abs to guard against invalid coords

def rect_intersect(rect1, rect2):
    step_x0, step_y0, step_x1, step_y1 = rect1
    tar_x0, tar_y0, tar_x1, tar_y1 = rect2

    if ((step_x0 <= tar_x0 <= step_x1 or step_x0 <= tar_x1 <= step_x1) and
            (step_y0 <= tar_y0 <= step_y1 or step_y0 <= tar_y1 <= step_y1)):
        return True
    return False

import math
from typing import Tuple

def rect_center_distance(rect1, rect2):
    # Unpack rectangle coordinates
    x0_1, y0_1, x1_1, y1_1 = rect1
    x0_2, y0_2, x1_2, y1_2 = rect2

    # Calculate center of rect1
    center1_x = (x0_1 + x1_1) / 2
    center1_y = (y0_1 + y1_1) / 2

    # Calculate center of rect2
    center2_x = (x0_2 + x1_2) / 2
    center2_y = (y0_2 + y1_2) / 2

    # Calculate Euclidean distance
    distance = math.sqrt((center2_x - center1_x) ** 2 + (center2_y - center1_y) ** 2)
    return distance


class StepProcessor:
    def __init__(self,top_folder,debug_level):

        self.top_folder = top_folder

        self.current_step_number = 1
        self.debug_level = debug_level

        self.page_size = None

        self.all_parts_df = None
        # self.pdf_document = None
        # self.pdf_page = None
        # self.steps_rects = None
        # self.step_font_size = caller.instruction_step_font_size
        # self.parts_font_size = caller.parts_font_size
        # self.numbered_substep_font_size = caller.numbered_substep_font_size

    def process_doc(self,pdf_doc,start_page_index,meta_dict,all_parts_df):

        page_processor = Page_Processor_v2.PageProcessor(debug_level= 1 )
        page_processor.set_meta(meta_dict)

        self.all_parts_df = all_parts_df

        # page 3 to page 40 - just for the beginning

        for page_index in range (start_page_index,396): #40


            page = pdf_doc[page_index]

            page_size = (int(page.rect.width), int(page.rect.height))

            page_processor.process_page(pdf_doc, page_index)
            step_blocks = self.__detect_steps_frame(page_processor,page_size)

            # if step_blocks == None:
            #     print ("Warning !!!! got None step block")
            #     continue

            # for each block we can create a folder
            for step_block in step_blocks:

                step_folder = os.path.join(self.top_folder, "s"+str(step_block.step_num))
                # print (step_folder)
                os.makedirs(step_folder, exist_ok=True)
                step_block.step_folder_path = step_folder

            # assuming here, we already parsed the global parts list:
            # now it's time to parse local parts list, and mark them as used

            for step_block in step_blocks:

                # get relevant texts and images
                relevant_texts = []
                relevant_images = []

                relevant_page_sub_module_text = []
                relevant_page_sub_module_images = []

                # check for text inside parts frame
                for text in page_processor.parts_list_texts:
                    if text.used: continue
                    if rect_intersect(step_block.step_frame, text.pos):
                        relevant_texts.append(text)

                # check for images inside parts frame
                for image in page_processor.images_list:
                    if image.used: continue
                    if rect_intersect(step_block.step_frame, image.pos):
                        relevant_images.append(image)

                # check for page sub step texts
                for text in page_processor.numbered_sub_module_texts:
                    if text.used: continue
                    if rect_intersect(step_block.step_frame, text.pos):
                        relevant_page_sub_module_text.append(text)

                # check for framed sub step image
                # looks like there is a lot of noise in the steps image. maybe it's a background fix ?
                # it almost always near the submodule step number
                for image in relevant_images:
                    if image.used: continue

                    #if image is inside framed submodule, skip
                    image_found = False
                    for sub_frame in page_processor.framed_sub_steps_list:
                        if rect_intersect(sub_frame.pos,image.pos):
                            # print ("droping image: ",image.xref)
                            image_found = True

                    if image_found: continue

                    x0, y0, x1, y1 = image.pos
                    area = abs((x1 - x0) * (y1 - y0))
                    if area < 250: continue
                    #add only unused images (maybe we should exclude colored submodules)
                    relevant_page_sub_module_images.append(image)

                if len(relevant_texts) == 0 or len(relevant_images) == 0:
                    print("Warning, no text or images found in part list")
                    break

                # process parts list, create files and parts.csv
                self.__process_parts_list(relevant_texts,relevant_images,step_block.step_folder_path)

                # got to the hard part. get numbered sub step
                # first check if they are really exists, but it's step number

                sub_module_blocks = None
                if relevant_page_sub_module_text:
                    sub_module_blocks = self.__detect_page_submodule(relevant_page_sub_module_text,relevant_page_sub_module_images)

                # now we have to take care about framed sub modules:
                # if they do exists, and sub_module_block exists, so probably the frame is from one of the sub modules

                if page_processor.framed_sub_steps_list and sub_module_blocks:
                    # ok, we have a framed sub module in page sub module
                    # lets assume for now, that a pages sub step will have only one sub frame
                    print ("page_processor.sub_steps_list ", page_processor.framed_sub_steps_list)
                    print ("sub_module_blocks: ",sub_module_blocks)


                    for sub_frame in page_processor.framed_sub_steps_list:
                        min_dist = 10000
                        min_page_step = None
                        for page_step in sub_module_blocks:
                            dist = rect_center_distance(sub_frame.pos, page_step.step_frame)
                            if min_dist > dist:
                                min_dist = dist
                                min_page_step = page_step
                        print ("found pair to ",min_page_step.step_num)


                ## don't forget to write to a file unused texts and images. they may help later. or indicate a problem

                # get the final step picture. it's probably the big one
                sorted_images = sorted(relevant_images, key=get_area, reverse=True)

                if sorted_images[0].used == False:
                    image_name = f"final_step.{sorted_images[0].image_extension}"
                    image_path = os.path.join(step_block.step_folder_path, image_name)
                    sorted_images[0].used = True
                    with open(image_path, "wb") as img_file:
                        img_file.write(sorted_images[0].image_bytes)
                else:
                    print ("Warning, no big image found")

            # inreech steps !

            #for each step, we have to collect all it's artifacts

            self.__show_debug_frames(page_processor.page_image,step_blocks,sub_module_blocks)

            key = cv2.waitKey(0) & 0xFF
            if key == 27:  #  Esc to quit
                break

            # here the top level logic starts.
            # probably parts list detection, and step size should be moved to here


            # page_processor.step_texts.clear()
            # page_processor.sub_module_texts.clear()
            # page_processor.page_sub_module_texts.clear()
            # page_processor.parts_list_texts.clear()
            # page_processor.other_texts.clear()
            #
            # page_processor.images_list.clear()
            # page_processor.other_frames_list.clear()
            # page_processor.parts_list.clear()
            # page_processor.sub_steps_list.clear()
            # page_processor.rotate_icons_list.clear()
            #
            # page_processor.steps.clear()

            #basic check - for for the first stp

    #maybe this schould be called init_step_block
    # and another function that fill step blocks
    def __detect_steps_frame(self,page_processor,page_size):

        pdf_x, pdf_y = page_size
        step_blocks = []

        step_texts = page_processor.step_texts.copy()
        # page_sub_module_texts = page_processor.page_sub_module_texts.copy()

        parts_list = page_processor.parts_list.copy()

        # exclude some obvious cases

        # if there is no steps numbers - return empty array !!!!!!!!!1 maybe this is wrong

        if len(step_texts) == 0 : return None

        if len(step_texts) > 0:
            # means we have some real steps,
            # and we can check if there is parts list

            # sort the list, so the last step on the page, will be last on array
            step_texts.sort(key=lambda block: int(block.text))

            # step 1 - detect the top left point - start of step

            for step_number in step_texts:

                step_block = StepBlock(
                    type = "step",
                    step_num=int(step_number.text)
                )
                x1, y1, x2, y2 = step_number.pos

                # check if there is a part list above it
                for part_list in parts_list:
                    rx1, ry1, rx2, ry2 = part_list.pos
                    if rx1 < x1 + 10 < rx2 and ry1 < y1 - 20 < ry2:
                        y1 = ry1 - 1
                        x2 = rx2
                        step_block.parts_frame = part_list.pos
                        break

                # make a few pixels before the number inside step area
                x1 = x1 - 10
                step_block.step_frame = (x1, y1, x2, y2)
                step_blocks.append(step_block)

            # step 2 - detect the bottom down point of each step

            for i in range(len(step_blocks) - 1):
                x1, y1, x2, y2 = step_blocks[i].step_frame

                index_right = -1
                index_down = -1

                # first check if THE NEXT ONE is below
                xx1, yy1, xx2, yy2 = step_blocks[i+1].step_frame

                if y2 < yy1:
                    # print(" found down of NEXT")
                    index_down = i + 1

                for j in range(i + 1, len(step_blocks)):
                    xx1, yy1, xx2, yy2 = step_blocks[j].step_frame

                    if x2 < xx1 and index_right == -1:
                        # print(" found right neighbor")
                        index_right = j

                # update from relevant neighbor
                if index_right != -1:
                    xx1, yy1, xx2, yy2 = step_blocks[index_right].step_frame
                    x2 = xx1 - 5
                else:
                    x2 = pdf_x - 5

                if index_down != -1:
                    xx1, yy1, xx2, yy2 = step_blocks[index_down].step_frame
                    y2 = yy1 - 5
                else:
                    y2 = pdf_y - 5

                step_blocks[i].step_frame = (x1, y1, x2, y2)

            # update the last element size

            x1, y1, x2, y2 = step_blocks[-1].step_frame
            step_blocks[-1].step_frame = (x1, y1, pdf_x - 5, pdf_y - 5)

        return step_blocks

    def __process_parts_list(self,relevant_texts,relevant_images,step_path):

        # get amount of parts,and pictures
        image_counter = 0

        # create a file for parts list
        parts_file_path = os.path.join(step_path , 'step_parts.csv')
        parts_file = open(parts_file_path, 'w')
        parts_file.write("element_id,amount,part_num,color_id\n")

        for parts_text in relevant_texts:

            if 'x' in parts_text.text:
                parts_text.used = True

                # here we should get the images of required parts
                tx0, ty0, tx1, ty1 = parts_text.pos
                image_counter += 1

                for each_image in relevant_images:
                    x0, y0, x1, y1 = each_image.pos
                    if x0 < tx0 + 5 < x1 and y0 < ty0 - 5 < y1:
                        image_name = f"prt_{image_counter}_{parts_text.text}.{each_image.image_extension}"
                        image_path = os.path.join(step_path, image_name)
                        each_image.used = True
                        with open(image_path, "wb") as img_file:
                            img_file.write(each_image.image_bytes)

                        # here we want to find each step element_id, and part_id
                        nparr_step = np.frombuffer(each_image.image_bytes, np.uint8)
                        img_in_step = cv2.imdecode(nparr_step, cv2.IMREAD_COLOR)

                        max_score = -2
                        max_matching_part = None

                        for index, row in self.all_parts_df.iterrows():
                            img_part = row['image']
                            if img_part is not None:
                                score = self.__compare_images_ssim(img_part, img_in_step)
                                if score > max_score:
                                    max_score = score
                                    max_matching_part = row

                        if max_matching_part is not None:
                            # print(f"Part {max_matching_part['element_id']}, score {max_score}")

                            if max_score > min_acceptance_score:
                                amount = parts_text.text.replace('x','')
                                parts_file.write(f"{max_matching_part['element_id']},{amount},"
                                                 f"{max_matching_part['part_num']},{max_matching_part['color_id']},\n")
                            else:
                                print ("WARNING: part match failed with score ",max_score)

                            # cv2.imshow('img_part', max_matching_part['image'])
                            # cv2.imshow('img_step', img_step)
                            # cv2.waitKey(0)
                            # cv2.destroyAllWindows()


                        continue

        parts_file.close()
        return None

    def __compare_images_ssim(self,img1, img2):
        # Resize img2 to match img1
        img2_resized = cv2.resize(img2, (img1.shape[1], img1.shape[0]))

        # Convert to grayscale
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(img2_resized, cv2.COLOR_BGR2GRAY)

        # Calculate SSIM
        score, _ = ssim(gray1, gray2, full=True)
        return score

    def __detect_page_submodule(self, relevant_text,relevant_images):

        sum_module_steps = []

        # sort the list, so the last step on the page, will be last on array
        relevant_text.sort(key=lambda block: int(block.text))

        # print ("submodule relevant text: ",relevant_text)

        '''
        lets try next approach:
        we have sorted step numbers. let's find the closes image to it, and remove it from list
        because we are starting from the begining, number 2 will have only one option (because we removed number 1)
        
        '''

        for step_text in relevant_text:
            #find the closest image

            min_dist = 100000
            min_image = None

            for image in relevant_images:

                if image.used: continue

                dist = rect_center_distance(step_text.pos,image.pos)
                if min_dist > dist:
                    min_dist = dist
                    min_image = image

            min_image.used = True

            # print ("text: ",step_text.text)
            # print ("min_dist: ",min_dist)
            # print("xref: ", min_image.xref)
            # print(" -------- \n\n")

            tx,ty,tx1,ty1 = step_text.pos
            ix,iy,ix1,iy1 = min_image.pos

            page_sub_module_block = StepBlock(
                type="page_submodule",
                step_num=int(step_text.text),
                step_frame = (min(tx,ix),min(ty,iy), max(tx1,ix1), max(ty1,iy1))
            )

            sum_module_steps.append(page_sub_module_block)
        return sum_module_steps


    def __fill_steps(self,step_blocks):
        '''



        now, we probably should collect all step belongings:
        1)

        '''


        pass

    def __show_debug_frames(self, page_image,step_blocks,sub_module_blocks):
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

            # # draw sub step frames
            # for frame in self.sub_steps_list:
            #     x0, y0, x1, y1 = frame.pos
            #     cv2.rectangle(working_page_basic, (x0, y0), (x1, y1), (255, 0, 255), 2)
            #
            # # draw parts frames
            # for frame in self.parts_list:
            #     x0, y0, x1, y1 = frame.pos
            #     cv2.rectangle(working_page_basic, (x0, y0), (x1, y1), (0, 255, 0), 2)
            #
            # # draw other frames
            # for frame in self.other_frames_list:
            #     x0, y0, x1, y1 = frame.pos
            #     cv2.rectangle(working_page_basic, (x0, y0), (x1, y1), (255, 0, 0), 2)
            #
            # # draw rotate icon
            # for rotate_icon in self.rotate_icons_list:
            #     x0, y0, x1, y1 = rotate_icon
            #     cv2.rectangle(working_page_basic, (x0, y0), (x1, y1), (0, 0, 255), 2)

            # draw steps
            for step in step_blocks:
                x0, y0, x1, y1 = step.step_frame
                cv2.rectangle(working_page_basic, (x0, y0), (x1, y1), (128, 128, 0), 2)

            if sub_module_blocks:
                for sub_module in sub_module_blocks:
                    x0, y0, x1, y1 = sub_module.step_frame
                    cv2.rectangle(working_page_basic, (x0, y0), (x1, y1), (0, 128, 128), 2)

            cv2.imshow("basic_debug", working_page_basic)

        cv2.waitKey(1)

#
#
# @dataclass
# class ContentBlock:
#     pos: Tuple[int, int, int, int]                      # (x1, y1, x2, y2)
#     type: Literal["image", "text"]                      # 'image' or 'text'
#     image_extension: Optional[str] = None               # e.g., 'png', 'jpg'
#     xref: Optional[str] = None
#     text: Optional[str] = None                          # extracted or associated text
#     font: Optional[int] = None
#     image_data: Optional[bytes] = None                  # binary image data
#     used: bool = False  # Default to unused
#
# def get_area(cb: ContentBlock) -> int:
#     x1, y1, x2, y2 = cb.pos
#     return abs((x2 - x1) * (y2 - y1))  # abs to guard against invalid coords
#
#

# min_acceptance_score = 0.8
# class StepProcessor:
#     def __init__(self,caller):
#         self.submodules_rects = None
#         self.parts_df = None
#
#         self.pdf_document = None
#         self.pdf_page = None
#         self.steps_rects = None
#         self.step_font_size = caller.instruction_step_font_size
#         self.parts_font_size = caller.parts_font_size
#         self.numbered_substep_font_size = caller.numbered_substep_font_size
#
#
#
#

#
#     def extract_step(self,top_folder,pdf_document, page_number, steps_rects,parts_df,submodules_rects):
#         self.pdf_document = pdf_document
#         self.pdf_page = pdf_document[page_number]
#         self.steps_rects = steps_rects
#         self.parts_df = parts_df
#         self.submodules_rects = submodules_rects
#
#         # print ("steps: ",steps_rects)
#         # print ("submodules: ",submodules_rects)
#
#         parts_file = None
#
#         for step in self.steps_rects:
#             texts = self.__extract_relevant_text(step)
#             images = self.__extract_relevant_images(step)
#
#             #get relevant submodules (they are rects, not objects)
#             submodules = []
#             for each_submodule in submodules_rects:
#                 if rect_intersect(step, each_submodule):
#                     submodules.append(each_submodule)
#
#             print ("submodules: ",submodules)
#             '''
#             in each step may be more than one submodule. and they may not have numbers
#             actualy, they will not have numbers if they are rounded submodules
#             '''
#
#
#
#             # nado proverit' ot kakogo shaga etot sub modul.
#             # potomu chto na stranitze est' 2 shaga. no submodule tol'ko v odnom iz nih
#             #
#             # kstati. na stranitze mogut bit 2 sabmodula ot dvuh raznih shagov. a mogut bit i ot odnogo
#             #
#             # kak bi tak sdelat' chtobi submodule obrabativalsya takje kak i prosto modul
#
#             # if self.submodules_rects:
#             #     submodule_images = self.__extract_relevant_submodule_images(submodules_rects[0])
#             #     print ("submodule_images",submodule_images)
#
#             # !!!!!!!!!!!1__extract_relevant_sub_module
#
#             step_number = 0
#             image_counter = 0
#
#             # !!! somehow, we should address sub,and sub-sub steps here
#             # and do remove used text and images, so they don't interfer'
#
#             # get step number, used for the folder name
#             for text in texts:
#                 if text.font_size == self.step_font_size:
#                     step_number = text.text
#                     folder = 's' + str(step_number)
#                     folder = os.path.join(top_folder, folder)
#                     os.makedirs(folder, exist_ok=True)
#
#                     # create a file for used parts list
#                     parts_file_path = os.path.join(folder, 'step_parts.csv')
#                     parts_file = open(parts_file_path, 'w')
#                     parts_file.write("element_id,amount,part_num,color_id\n")
#                     break
#
#             print(f"step number is: {step_number} , texts: {len(texts)} , images: {len(images)}")
#
#
#             # get amount of parts,and pictures
#             for text in texts:
#                 if text.font_size == self.parts_font_size and 'x' in text.text:
#                     # here we should get the images of required parts
#                     tx0, ty0, tx1, ty1 = text.pos
#                     image_counter += 1
#
#                     for each in images:
#                         x0, y0, x1, y1 = each.pos
#                         if x0 < tx0 + 5 < x1 and y0 < ty0 - 5 < y1:
#                             image_name = f"prt_{image_counter}_{text.text}.{each.image_extension}"
#                             image_path = os.path.join(folder, image_name)
#                             each.used = True
#                             with open(image_path, "wb") as img_file:
#                                 img_file.write(each.image_data)
#
#                             #here we want to find each step element_id, and part_id
#                             nparr_step = np.frombuffer(each.image_data, np.uint8)
#                             img_step = cv2.imdecode(nparr_step, cv2.IMREAD_COLOR)
#
#                             max_score = -2
#                             max_matching_part = None
#
#                             for index, row in self.parts_df.iterrows():
#                                 img_part = row['image']
#                                 if img_part is not None:
#                                     score = self.__compare_images_ssim(img_part, img_step)
#                                     if score > max_score:
#                                         max_score = score
#                                         max_matching_part = row
#
#                             if max_matching_part is not None:
#                                 # print(f"Part {max_matching_part['element_id']}, score {max_score}")
#
#                                 if max_score > min_acceptance_score:
#                                     amount = text.text.replace('x','')
#                                     parts_file.write(f"{max_matching_part['element_id']},{amount},"
#                                                      f"{max_matching_part['part_num']},{max_matching_part['color_id']},\n")
#                                 else:
#                                     print ("part match failed with score ",max_score)
#
#                                 # cv2.imshow('img_part', max_matching_part['image'])
#                                 # cv2.imshow('img_step', img_step)
#                                 # cv2.waitKey(0)
#                                 # cv2.destroyAllWindows()
#
#
#                             continue
#
#             parts_file.close()
#
#             # get the biggest image. it's probably the step image
#             sorted_images = sorted(images, key=get_area, reverse=True)
#
#             image_name = f"big_step.{sorted_images[0].image_extension}"
#             image_path = os.path.join(folder, image_name)
#             sorted_images[0].used = True
#             with open(image_path, "wb") as img_file:
#                 img_file.write(sorted_images[0].image_data)
#
#
#
#
#
#             # here we left with unused images, that are probably from the submodule
#             for i,submodule in enumerate(submodules):
#
#                 '''
#                 up to here, we parsed the main step ----------- start submodule
#                 rounded submodule can be with with numbers, but also can be without.
#                 and can be with '2x'.
#
#                 if there is _x, it probably should be in submodule name
#                 '''
#
#                 submodule_texts = self.__extract_relevant_text(submodule)
#                 submodule_multiplier = "1x"
#                 if submodule_texts:
#                     # look for _x text
#                     for text in submodule_texts:
#                         if 'x' in text.text:
#                             submodule_multiplier = text.text
#
#
#
#                 # i want to use the same images array, so we left with true unused images
#                 submodule_images = []
#                 for image in sorted_images:
#                     image_rec = image.pos
#                     if rect_intersect(submodule, image_rec):
#                         submodule_images.append(image)
#
#
#                 #create subfolder
#                 subfolder_path = os.path.join(folder, "sub"+str(i)+"_"+submodule_multiplier)
#                 os.makedirs(subfolder_path, exist_ok=True)
#
#                 #write all images here
#                 image_counter = 0
#                 for each in submodule_images:
#
#                     image_name = f"sub_{image_counter}.{each.image_extension}"
#                     image_counter += 1
#                     image_path = os.path.join(subfolder_path, image_name)
#                     each.used = True
#                     with open(image_path, "wb") as img_file:
#                         img_file.write(each.image_data)
#
#                 remarks = []
#                 for each in submodule_texts:
#                     remarks.append(each.text)
#
#                 if remarks:
#                     text = ', '.join(remarks)
#                     image_path = os.path.join(subfolder_path, "remarks.txt")
#                     with open(image_path, "w") as txt_file:
#                         txt_file.write(text)
#
#
#             # get all unused images
#             image_counter = 0
#             for each in sorted_images:
#                 if each.used == False:
#                     image_name = f"imgUU_{image_counter}.{each.image_extension}"
#                     image_path = os.path.join(folder, image_name)
#                     each.used = True
#                     image_counter += 1
#
#                     with open(image_path, "wb") as img_file:
#                         img_file.write(each.image_data)
#
#
#
#
#
