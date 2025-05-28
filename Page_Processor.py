'''

the idea of this class is like this:
it's going to take a page from lego instructions, and get some data
for now i know two types:
1) the text embeded in pdf files
2) rounded squares for building parts, and submodules

so actually this is a place for some logic and computer vision
'''


import numpy as np
import cv2
import warnings



def is_inside_rect(px,py, rect):
    rx1,ry1,rx2,ry2 = rect
    return rx1 < px < rx2 and ry1 < py < ry2


class PageProcessor:
    def __init__(self, debug = False):
        self.debug = debug
        self.parts_list_color = None
        self.sub_step_color = None

    def set_parts_list_color(self,color):
        self.parts_list_color = color

    def set_sub_step_color(self,color):
        self.sub_step_color = color

    def get_meta(self):
        meta = {}
        meta["parts_list_color"] = self.parts_list_color
        meta["sub_step_color"] = self.sub_step_color
        meta["irrelevant_pages"] = self.irrelevant_pages
        meta["step_font_size"] = 28



    '''
    this functions detects parts list candidates by color.
    then, it chackes if there is some text with numX (the list created at pdf parsing)
    if we do found some "x" number inside the candidate - so it's a part list
    
    theoreticlym it's posible to automate it by detecting the numX with font 8
    ,extracting background, and applying color filter. TBD
    '''
    def detect_parts_list(self,cv_image,parts_list_numbers):

        validated_parts_list = []

        if self.parts_list_color is None:
            warnings.warn("parts list color must be set")
            return None
        elif not parts_list_numbers:
            return None
        else:
            parts_list_candidates = self.__get_sub_window_locations(cv_image,self.parts_list_color)
            for parts_list_candidate in parts_list_candidates:
                x1, y1, x2, y2 = parts_list_candidate
                #print("candidate: ", parts_list_candidate)
                # we need candidate for 1:1 window, rotation sign, and sub sub module

                for txt,parts_list_number_pose in parts_list_numbers:
                    x1x, y1x, x2x, y2x = parts_list_number_pose
                    if x1 < x1x < x2 and y1 < y1x < y2 :
                        validated_parts_list.append(parts_list_candidate)

            return validated_parts_list



    def detect_sub_steps(self,cv_image):
        if self.sub_step_color is None:
            warnings.warn("sub set color must be set")
            return None
        else:
            return self.__get_sub_window_locations(cv_image,self.sub_step_color)

    def __get_sub_window_locations(self,cv_image, color, tolerance = 10):

        # Step 1: calc color bounding
        color_int = color.astype(int)
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
                rects.append((x,y,x+w,y+h))

        # Draw rectangles on a copy of the original image
        if self.debug:
            # mask applied
            masked_image = cv_image.copy()
            result = cv2.bitwise_and(masked_image, masked_image, mask=mask)

            # detected rectangles
            rects_image = cv_image.copy()
            for rect in rects:
                x1, y1, x2, y2 = rect
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

    def detect_steps_area(self,steps_number_list,parts_list,page_size):

        pdf_x, pdf_y = page_size
        steps = []

        # exclude some obvious cases

        # if there is no steps numbers - return empty array
        length = len(steps_number_list)
        if length == 0: return steps


        # step 1 - detect the top left point - start of step

        for txt, (x1,y1,x2,y2) in steps_number_list:
        
            # check if there is a part list above it
            if parts_list:
                for rect in parts_list:
                    rx1, ry1, rx2, ry2 = rect
                    if rx1 < x1+10 < rx2 and ry1 < y1-10 < ry2:
                        y1 = rect[1] - 1
                        break

            # make a few pixels before the number inside step area
            x1 = x1 - 10
            steps.append((x1,y1,x2,y2))


        # if we have only one step
        if length == 1:
            x1, y1, x2, y2 = steps[0]
            steps[0] = (x1, y1, pdf_x - 5, pdf_y - 5)
            return steps

        # step 2
        for i in range (len(steps)-1):
            x1, y1, x2, y2 = steps[i]

            index_right = -1
            index_down = -1

            # first check if HE NEXT ONE is below
            xx1, yy1, xx2, yy2 = steps[i+1]

            if y1 < yy1 :
                # print(" found down of NEXT")
                index_down = i + 1

            for j in range(i+1,len(steps)):
                xx1, yy1, xx2, yy2 = steps[j]

                if x1 < xx1 and index_right == -1:
                    # print(" found right neighbor")
                    index_right = j


            # update from relevant neighbor
            if index_right != -1:
                xx1, yy1, xx2, yy2 = steps[index_right]
                x2 = xx1 - 5
            else:
                x2 = pdf_x - 5

            if index_down != -1:
                xx1, yy1, xx2, yy2 = steps[index_down]
                y2 = yy1 - 5
            else:
                y2 = pdf_y - 5

            steps[i] = (x1, y1, x2, y2)

        # update the last element size

        x1, y1, x2, y2 = steps[-1]
        steps[-1] = (x1, y1, pdf_x-5, pdf_y-5)

        return steps






















