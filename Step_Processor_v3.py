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
import warnings

import fitz  # PyMuPDF
from dataclasses import dataclass
from typing import Literal, Optional, Tuple, List, Union, Self
import math

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

'''
let's try a new approach:
there is only two variants:
a step, and it's sub_step. 

a sub_step can include more substeps, and so on.
no diferences between numbered sub steps, and framed sub steps (or sub sub steps)
a uniqe sub step can have a step number of zero
'''

min_acceptance_score = 0.8

@dataclass
class StepBlock:
    type: Literal["step", "sub_step"]
    step_num: int
    step_frame: Optional[Tuple[int, int, int, int]] = None
    parts_frame: Optional[Tuple[int, int, int, int]] = None
    step_folder_path: Optional[str] = None

    multiplier: int = 1
    rotate: bool = False
    used: bool = False
    # a list of sub steps
    sub_steps: Optional[List["StepBlock"]] = None  # Self-reference here

    final_image_block: Optional["ContentBlock"] = None




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

def rect_inside(outer_rect, inner_rect):
    out_x0, out_y0, out_x1, out_y1 = outer_rect
    in_x0, in_y0, in_x1, in_y1 = inner_rect

    if out_x0 <= in_x0 <= in_x1 <= out_x1 and out_y0 <= in_y0 <= in_y1 <= out_y1:
        return True
    return False

def rect_center_distance(rect1, rect2):
    # Unpack rectangle coordinates
    x0_1, y0_1, x1_1, y1_1 = rect1
    x0_2, y0_2, x1_2, y1_2 = rect2

    # Calculate center of rect1
    center1_x = (x0_1 + x1_1) / 2
    center1_y = (y0_1 + y1_1) / 2

    # Calculate center of rect2
    # center2_x = (x0_2 + x1_2) / 2
    # center2_y = (y0_2 + y1_2) / 2
    center2_x = x0_2
    center2_y = y0_2

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


    def process_doc(self,pdf_doc,start_page_index,meta_dict,all_parts_df):

        # some inits
        page_processor = Page_Processor_v2.PageProcessor(debug_level=0)
        page_processor.set_meta(meta_dict)

        self.all_parts_df = all_parts_df

        total_step_bocks = []

        '''
        
        the methods goes like this:
        process a page:
            1) get steps from it
            2) update parts details
            3) update NUMBERED sub steps
            
            check results
            4) if there is numbered substeps, update each one with framed sub step
            5) if no numbered substeps, create them from framed sub steps
            
            6) done ?
         
        '''
        # start_page_index = 12
        for page_index in range (start_page_index,396): #40

            # get a page
            page = pdf_doc[page_index]
            page_size = (int(page.rect.width), int(page.rect.height))

            # process page, get steps
            page_processor.process_page(pdf_doc, page_index)
            step_blocks = self.__detect_steps_frame(page_processor,page_size)




            # tmp skip, while working on sub steps
            # if page_processor.numbered_sub_module_texts is None or len(page_processor.numbered_sub_module_texts) == 0:
            #     continue

            # if not page_processor.framed_sub_steps_list:
            #     continue

            if step_blocks is None:
                # here is a uniq case, where no steps exist. generaly it's a final part of numbered substeps
                print ("Warning !!!! got None step block")

                # rotation angle may be here too
                # rotate_icons = page_processor.rotate_icons_list
                # print("rotate_icons: ", rotate_icons)

                if page_processor.numbered_sub_module_texts:
                    print ("we do hove some numbered block")

                    # the sub block may contain framed block
                    # or there maybe more that one

                    # first check for framed step
                    # check for framed sub steps
                    # we have to do it in the step loop, so relevant images will be relevant
                    numbered_step_block = StepBlock(
                        type="sub_step",
                        step_num=0, #!!!!!!!!!!!!!1 change here
                        # this should be set properly, if there is do step on the page (but how to know it ?)
                        step_frame = (1, 1, int(page.rect.width), int(page.rect.height))
                    )

                    # for each step area, check if there is numbered sub steps
                    numbered_steps = self.__detect_page_submodule(page_processor, numbered_step_block, page_processor.images_list)

                    # get framed sub step
                    framed_sub_steps = []
                    if page_processor.framed_sub_steps_list:
                        for frame in page_processor.framed_sub_steps_list:

                            if not numbered_step_block.sub_steps:
                                step_block.sub_steps = []

                            sub_frame = self.__process_frame(page_processor, frame, page_processor.images_list)
                            framed_sub_steps.extend(sub_frame)

                        numbered_steps[0].sub_steps = framed_sub_steps

                    # process the numbered block, add framed by the end



                    numbered_step_block.sub_steps = numbered_steps

                    print ("last real step block: ",total_step_bocks[-1])
                    print ("stand alone numbered_step_block: ",numbered_step_block)
                    self.pretty_print_stepblock(numbered_step_block)

                # continue

                step_blocks = []

            # proceed each parts list of the step
            for step_block in step_blocks:

                # get relevant texts and images
                relevant_texts = []
                relevant_images = []

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

                # process parts list, create files and parts.csv
                step_block.step_folder_path = os.path.join(self.top_folder, "s" + str(step_block.step_num))
                parts_meta,parts_images = self.__process_parts_list(relevant_texts, relevant_images)

                # after we have extracted small parts, filter images with small area
                # i'm marking them as used, so other logic will skip it (hopefully)
                for image in relevant_images:
                    x0, y0, x1, y1 = image.pos
                    area = abs((x1 - x0) * (y1 - y0))
                    if area < 500:
                        image.used = True

                # ---------------------------------------------------
                # up to here we have a list of steps, with their area
                #----------------------------------------------------

                # check for framed sub steps
                # we have to do it in the step loop, so relevant images will be relevant
                framed_sub_steps = []
                if page_processor.framed_sub_steps_list:
                    for frame in page_processor.framed_sub_steps_list:
                        if rect_intersect(step_block.step_frame,frame.pos):
                            if not step_block.sub_steps:
                                step_block.sub_steps = []

                            sub_frame = self.__process_frame(page_processor,frame,relevant_images)
                            framed_sub_steps.extend(sub_frame)


                # for each step area, check if there is numbered sub steps
                numbered_steps = []
                if page_processor.numbered_sub_module_texts:
                    numbered_steps = self.__detect_page_submodule(page_processor,step_block,relevant_images)

                # do some logic, to place the sub steps (numbered or framed) into the step

                #only numbered steps
                if not framed_sub_steps and numbered_steps:
                    step_block.sub_steps = numbered_steps
                # only framed steps
                elif framed_sub_steps and not numbered_steps:
                    step_block.sub_steps = framed_sub_steps
                # both of them
                else:
                    # here we have both numbered and sub.
                    # so we need to find for each frame, to what numbered step it belongs
                    for frame in framed_sub_steps:
                        min_dist = 10000
                        min_numbered_step = None
                        for numbered_step in numbered_steps:
                            dist = rect_center_distance(numbered_step.step_frame, frame.step_frame)
                            if dist < min_dist:
                                min_dist = dist
                                min_numbered_step = numbered_step

                        if min_numbered_step.sub_steps is None:
                            min_numbered_step.sub_steps = []
                        min_numbered_step.sub_steps.append(frame)

                    step_block.sub_steps = numbered_steps

                # check for rotation icons
                if page_processor.rotate_icons_list:
                    for rotation_icon in page_processor.rotate_icons_list:
                        self.__insert_rotation_flag(step_block,rotation_icon)

                # search for 1:1 frames, and mark all inside images as used
                self.__clean_1_1_frames(page_processor,relevant_images)

                #if only one unused image left, it's probably the final image of the step
                #if mor that one image left - rise a warning
                # if not even one left, we probably in numbered step
                left_images_count = 0
                last_image = None

                sorted_images = sorted(relevant_images, key=get_area, reverse=False)

                for image in sorted_images:
                    if image.used is False:
                        left_images_count += 1
                        last_image = image

                if left_images_count == 1: # it's the final step image
                    step_block.final_image_block = last_image
                    last_image.used = True
                else:
                    # here we are checking if some image left for STEP (not whole page
                    print(f"step {step_block.step_num} , images left: {left_images_count}")


                # save processed block to full blocks list (hopefully we have enough RAM)

                total_step_bocks.append(step_block)

            # check for page if something left
            for text in page_processor.numbered_sub_module_texts:
                if text.used is False:
                    print ("we have left some text: a numbered submodule !!!!! ")

            counter = 0
            for image in page_processor.images_list:
                if image.used is False:
                    counter += 1

            if counter > 0:
                print("we have left some images on the page  !!!!! ",counter)

            # here goes a verification function.
            # we should check for the page, that there is no images left
            # for now, for sure there are: but we have to dela with it


            # print ("step is",step_block.step_num)
            self.__show_unused_images(page_processor.page_image,page_processor.images_list)
            self.__show_debug_frames(page_processor.page_image,step_blocks)

            key = cv2.waitKey(0) & 0xFF
            if key == 27:  #  Esc to quit
                break


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

    def __process_parts_list(self,relevant_texts,relevant_images):

        '''
        the result of this function is:
        1) a list of images (object) that includes image name, sufix,and data
        2) a dict that contains tuples of (element_id,amount,part_num,color_id)

        '''

        # define internal functions, as it's used only here

        def __compare_images_ssim(img1, img2):
            # Resize img2 to match img1
            img2_resized = cv2.resize(img2, (img1.shape[1], img1.shape[0]))

            # Convert to grayscale
            gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
            gray2 = cv2.cvtColor(img2_resized, cv2.COLOR_BGR2GRAY)

            # Calculate SSIM
            score, _ = ssim(gray1, gray2, full=True)
            return score

        images_list = []
        images_meta_dict = []

        # get amount of parts,and pictures
        image_counter = 0

        for parts_text in relevant_texts:

            if 'x' in parts_text.text:
                parts_text.used = True

                # here we should get the images of required parts
                tx0, ty0, tx1, ty1 = parts_text.pos
                image_counter += 1

                for each_image in relevant_images:
                    x0, y0, x1, y1 = each_image.pos
                    if x0 < tx0 + 5 < x1 and y0 < ty0 - 5 < y1:
                        each_image.image_name = f"prt_{image_counter}_{parts_text.text}.{each_image.image_extension}"
                        each_image.used = True

                        images_list.append(each_image)
                        # here we want to find each step element_id, and part_id
                        nparr_step = np.frombuffer(each_image.image_bytes, np.uint8)
                        img_in_step = cv2.imdecode(nparr_step, cv2.IMREAD_COLOR)

                        max_score = -2
                        max_matching_part = None

                        for index, row in self.all_parts_df.iterrows():
                            img_part = row['image']
                            if img_part is not None:
                                score = __compare_images_ssim(img_part, img_in_step)
                                if score > max_score:
                                    max_score = score
                                    max_matching_part = row

                        if max_matching_part is not None:
                            # print(f"Part {max_matching_part['element_id']}, score {max_score}")

                            if max_score > min_acceptance_score:
                                amount = parts_text.text.replace('x','')

                                images_meta_dict.append((
                                    max_matching_part['element_id'],amount,
                                    max_matching_part['part_num'],max_matching_part['color_id']))

                            else:
                                print ("WARNING: part match failed with score ",max_score)

                                # cv2.imshow('img_part', max_matching_part['image'])
                                # cv2.imshow('img_in_step', img_in_step)
                                # cv2.waitKey(0)
                                # cv2.destroyAllWindows()

                        continue
        return images_meta_dict,images_list

    def __detect_page_submodule(self, page_processor,step_block,images):

        sum_module_steps = []

        # filter only needed text and images

        relevant_text = []
        relevant_images = []

        # check for relevant page sub step texts
        for text in page_processor.numbered_sub_module_texts:
            if text.used: continue
            if rect_intersect(step_block.step_frame, text.pos):
                relevant_text.append(text)

        # check for relevant images
        for image in images:
            if image.used: continue
            if rect_intersect(step_block.step_frame, image.pos):
                relevant_images.append(image)

        # sort the list, so the last step on the page, will be last on array
        relevant_text.sort(key=lambda block: int(block.text))

        '''
        lets try next approach:
        we have sorted step numbers. let's find the closes image to it, and remove it from list
        because we are starting from the begining, number 2 will have only one option (because we removed number 1)
        '''

        for step_text in relevant_text:
            min_dist = 100000
            min_image = None
            for image in relevant_images:
                if image.used: continue
                dist = rect_center_distance(step_text.pos,image.pos)
                if min_dist > dist:
                    min_dist = dist
                    min_image = image
            min_image.used = True
            step_text.used = True

            # print ("text: ",step_text.text)
            # print ("min_dist: ",min_dist)
            # print("xref: ", min_image.xref)
            # print(" -------- \n\n")

            tx,ty,tx1,ty1 = step_text.pos
            ix,iy,ix1,iy1 = min_image.pos

            page_sub_module_block = StepBlock(
                type="sub_step",
                step_num=int(step_text.text),
                step_frame = (min(tx,ix),min(ty,iy), max(tx1,ix1), max(ty1,iy1)),
                final_image_block = min_image
            )

            sum_module_steps.append(page_sub_module_block)
        return sum_module_steps

    def __process_frame(self,page_processor,frame,relevant_images):
        '''
        consider each frame as a small page.
        it may have steps, images,multiplier, and sub step
        start with text. if no text, so there is only one step with one image
        '''
        frame_blocks = []

        # get relevant text
        frame_relevant_texts = []
        frame_relevant_images = []

        # start with images, they exist for sure
        for image in relevant_images:

            if image.used: continue

            # x0, y0, x1, y1 = image.pos
            # area = abs((x1 - x0) * (y1 - y0))
            # if area < 250: continue

            if rect_inside(frame.pos, image.pos):
                frame_relevant_images.append(image)

        frame_multiplier = 1
        for text in page_processor.frame_sub_module_texts:
            if rect_inside(frame.pos, text.pos):
                # a unique case: _x can be in frame. we can assume only one text of this form
                if 'x' in text.text:
                    frame_multiplier = int(text.text[:-1])
                else:
                    frame_relevant_texts.append(text)

        # check for sub sub frame
        sub_sub_block = None
        for sub_frame in page_processor.other_frames_list:
            if rect_inside(frame.pos,sub_frame.pos):
                sub_sub_block = self.__process_sub_sub_block(sub_frame,frame_relevant_images)
                sub_frame.used = True

        if not frame_relevant_texts: # means no text in frame
            # 1) just get the image
            # 2) store it on object

            if len(frame_relevant_images) > 1:
                warnings.warn("frame has multiple images. expected ONLY one")

            frame_sub_step_block = StepBlock(
                type="sub_step",
                step_num=0,
                final_image_block = frame_relevant_images[0],
                step_frame=frame.pos,
                multiplier = frame_multiplier
                )

            frame_relevant_images[0].used = True

            if sub_sub_block:
                frame_sub_step_block.sub_steps = [sub_sub_block]

            frame_blocks.append(frame_sub_step_block)
            return frame_blocks

        else:
            # means there is texts, so it's a multi step frame

            # add an empty sub frame. it should hepl in case there is two framed steps in main step.
            # and some of frames has internal order, because they are multi step

            # it mau have a multiplier ??
            intermediate_list = []
            intermediate_block = StepBlock(
                type="sub_step",
                step_num=-1,
                step_frame=frame.pos
                # multiplier=frame_multiplier
            )

            #sort text
            frame_relevant_texts.sort(key=lambda block: int(block.text))

            #for each number find the closest image
            for step_text in frame_relevant_texts:
                # find the closest image

                min_dist = 100000
                min_image = None

                for image in frame_relevant_images:

                    if image.used: continue

                    dist = rect_center_distance(step_text.pos, image.pos)
                    if min_dist > dist:
                        min_dist = dist
                        min_image = image

                min_image.used = True

                # print ("text: ",step_text.text)
                # print ("min_dist: ",min_dist)
                # print("xref: ", min_image.xref)
                # print(" -------- \n\n")

                tx, ty, tx1, ty1 = step_text.pos
                ix, iy, ix1, iy1 = min_image.pos

                frame_sub_step_block = StepBlock(
                    type="sub_step",
                    step_num=int(step_text.text),
                    step_frame=(min(tx, ix), min(ty, iy), max(tx1, ix1), max(ty1, iy1)),
                    final_image_block = min_image,
                    multiplier = frame_multiplier
                )

                # frame_blocks.append(frame_sub_step_block)
                intermediate_list.append(frame_sub_step_block)

                if sub_sub_block:
                    min_dist = 10000
                    min_frame = None
                    for frame in intermediate_list:
                        dist = rect_center_distance(frame.step_frame, sub_sub_block.step_frame)
                        # print("dist: ", dist)
                        if min_dist > dist:
                            min_dist = dist
                            min_frame = frame

                    # print("min_frame: ", min_frame)
                    min_frame.sub_steps = [sub_sub_block]

            # here we have all frame sub steps.
            # if we do have a sub sub step, we should find the closes sub step to it

            intermediate_block.sub_steps = intermediate_list
            frame_blocks.append(intermediate_block)

            return frame_blocks



        # check for multiplier

    def __process_sub_sub_block(self,sub_sub_frame, relevant_images):
        '''

         ok, so here is relevantly simple block. it has a frame, and maybe a few images
         if there is only one image - it's probably ok.
         if more - put a warning. its probably some unique case, that should be fixed manually

         may it has '2x' ? no idea. let's say that not, till i found it wrong

        '''
        sub_sub_images = []
        for image in relevant_images:
            if rect_inside(sub_sub_frame.pos,image.pos):
                image.used = True
                sub_sub_images.append(image)

        if len(sub_sub_images) > 1:
            warnings.warn("sub sub frame has multiple images. expected ONLY one")

        sub_sub_block = StepBlock(
            type="sub_step",
            step_num=int(0),
            step_frame=sub_sub_frame.pos,
            final_image_block = sub_sub_images[0]
        )

        return sub_sub_block

    def __insert_rotation_flag(self,step,rotation_location):
        '''
        the idea here is like this:
        go over step, adn each nested element.
        then from bottom up, check if it's found

        sound like recursion....
        '''


        if step.sub_steps:
            for sub_step in step.sub_steps:
                if sub_step.sub_steps:
                    for sub_step2 in sub_step.sub_steps:
                        if sub_step2.sub_steps:
                            for sub_step3 in sub_step2.sub_steps:
                                if rect_intersect(sub_step3.step_frame,rotation_location):
                                    sub_step3.rotate = True
                                    return

                        if rect_intersect(sub_step2.step_frame, rotation_location):
                            sub_step2.rotate = True
                            return

                    if rect_intersect(sub_step.step_frame, rotation_location):
                        sub_step.rotate = True
                        return

            if rect_intersect(step.step_frame, rotation_location):
                step.rotate = True
                return

    def __clean_1_1_frames(self,page_processor,relevant_images):
        # remove frames with 1:1. it can contain unused (for me) images. just mark them as used
        list_1_1 = []
        for text in page_processor.other_texts:
            if text.text == "1:1":
                list_1_1.append(text)
                text.used = True

        for other_frame in page_processor.other_frames_list:
            if other_frame.used: continue

            for text_1_1 in list_1_1:
                if rect_intersect(other_frame.pos, text_1_1.pos):
                    # mark it as used, to skip it in the final summary
                    # other_frame.used = True
                    # mark all images inside it as used
                    for image in relevant_images:
                        if image.used: continue
                        if rect_inside(other_frame.pos, image.pos):
                            image.used = True
                            other_frame.used = True


    def __show_unused_images(self, page_image,images):
        working_page_basic = page_image.copy()
        # counter = 0
        for image in images:
            if not image.used:
                # counter +=1
                x0, y0, x1, y1 = image.pos
                cv2.rectangle(working_page_basic, (x0, y0), (x1, y1), (128, 128, 0), 2)

        # print(f"what is left is {counter}")

        cv2.imshow("unuzed_images", working_page_basic)
        cv2.waitKey(1)


    def __show_debug_frames(self, page_image,step_blocks):
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

            # actually, we have a full step. it may have numbered steps, stat may have frame steps, that may have sub step
            # step -> numbered_step -> frame_step -> frame_sub_step
            # it's clearly recursive ,but I will make it quick and dirty, because the max depth is 'only' 4

            # draw steps
            for step in step_blocks:
                x0, y0, x1, y1 = step.step_frame
                cv2.rectangle(working_page_basic, (x0, y0), (x1, y1), (128, 128, 0), 2)

                if step.sub_steps:
                    for sub_step in step.sub_steps:
                        x0, y0, x1, y1 = sub_step.step_frame
                        cv2.rectangle(working_page_basic, (x0, y0), (x1, y1), (0, 255, 0), 2)

                        if sub_step.sub_steps:
                            for sub_step2 in sub_step.sub_steps:
                                x0, y0, x1, y1 = sub_step2.step_frame
                                cv2.rectangle(working_page_basic, (x0, y0), (x1, y1), (0, 128, 255), 2)

                                if sub_step2.sub_steps:
                                    for sub_step3 in sub_step2.sub_steps:
                                        x0, y0, x1, y1 = sub_step3.step_frame
                                        cv2.rectangle(working_page_basic, (x0, y0), (x1, y1), (0, 0, 255), 2)

                # self.pretty_print_stepblock (step)
                if step.final_image_block is not None:
                    x0, y0, x1, y1 = step.final_image_block.pos
                    cv2.rectangle(working_page_basic, (x0, y0), (x1, y1), (0, 0, 255), 3)
                else:
                    self.pretty_print_stepblock(step)
                                # and here also can be a sub sub module

            cv2.imshow("basic_debug", working_page_basic)
        cv2.waitKey(1)

    def pretty_print_stepblock(self,step, indent=0):
        ind = '    ' * indent
        print(f"{ind}StepBlock(")
        print(f"{ind}    type={repr(step.type)},")
        print(f"{ind}    step_num={step.step_num},")
        print(f"{ind}    step_frame={step.step_frame},")
        print(f"{ind}    parts_frame={step.parts_frame},")
        print(f"{ind}    step_folder_path={repr(step.step_folder_path)},")
        print(f"{ind}    multiplier={step.multiplier},")
        print(f"{ind}    rotate={step.rotate},")
        print(f"{ind}    used={step.used},")

        if step.sub_steps:
            print(f"{ind}    sub_steps=[")
            for sub in step.sub_steps:
                self.pretty_print_stepblock(sub, indent + 2)
            print(f"{ind}    ],")
        else:
            print(f"{ind}    sub_steps=None,")

        if step.final_image_block:
            block = step.final_image_block
            print(f"{ind}    final_image_block=ContentBlock(")
            print(f"{ind}        type={repr(block.type)},")
            print(f"{ind}        pos={block.pos},")
            print(f"{ind}        text={repr(block.text)},")
            print(f"{ind}        font_size={block.font_size},")
            print(f"{ind}        image_extension={repr(block.image_extension)},")
            print(f"{ind}        xref={block.xref},")
            print(f"{ind}        used={block.used}")
            print(f"{ind}    )")
        else:
            print(f"{ind}    final_image_block=None")

        print(f"{ind})")
