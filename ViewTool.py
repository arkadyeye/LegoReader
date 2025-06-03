import cv2
import numpy as np
import os

import Page_Processor_v2
import PdfHandler
import Step_Processor_v2
import Step_Processor_v3

#######
manuals_folder = "Manuals"

#######


zoom = 1
def show_zoomed_image(window_name, cv_image):
    height, width = cv_image.shape[:2]
    new_size = (int(width * zoom), int(height * zoom))
    resized = cv2.resize(cv_image, new_size, interpolation=cv2.INTER_CUBIC)
    cv2.imshow(window_name, resized)


class LegoInstructionParser:
    def __init__(self):
        self.window_name = "LEGO Instruction Parser"

        self.pdf_name = None
        self.pdf_handler = None

        self.selection_start = None
        self.selection_end = None
        self.drawing = False

        # vars for rectangle color selection
        self.ref_point = []
        self.cropping = False
        self.image = None
        self.clone = None
        self.preview_image = None

    def mouse_callback(self,event, x, y, flags, param):

        # anti-zoom mouse coordinates
        x = int(x/zoom)
        y = int(y/zoom)

        if event == cv2.EVENT_LBUTTONDOWN:
            self.ref_point = [(x, y)]
            self.cropping = True
            self.preview_image = self.pdf_handler.get_img()

        elif event == cv2.EVENT_MOUSEMOVE:
            if self.cropping:
                self.preview_image = self.pdf_handler.get_img()
                cv2.rectangle(self.preview_image, self.ref_point[0], (x, y), (0, 255, 0), 2)

                # cv2.imshow(self.window_name, self.preview_image)
                show_zoomed_image(self.window_name, self.preview_image)

        elif event == cv2.EVENT_LBUTTONUP:
            self.ref_point.append((x, y))
            self.cropping = False
            self.preview_image = self.pdf_handler.get_img()
            cv2.rectangle(self.preview_image, self.ref_point[0], self.ref_point[1], (0, 255, 0), 2)
            # cv2.imshow(self.window_name, self.preview_image)
            show_zoomed_image(self.window_name, self.preview_image)



    def show_page_decorations(self):
        working_page = self.pdf_handler.get_img()
        if working_page is None:
            return

        if self.pdf_handler.steps_numbers:
            for num,bbox in self.pdf_handler.steps_numbers:
                x0, y0, x1, y1 = bbox
                cv2.rectangle(working_page, (x0, y0), (x1, y1), (0, 255, 0), 2)
                # cv2.putText(img, str(i),  (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        if  self.pdf_handler.partslist_numbers:
            for num,bbox in self.pdf_handler.partslist_numbers:
                x0, y0, x1, y1 = bbox
                cv2.rectangle(working_page, (x0, y0), (x1, y1), (0, 255, 0), 2)

        if self.pdf_handler.parts_list:
            for bbox in self.pdf_handler.parts_list: # x,y,w,h
                x0, y0, x1, y1 = bbox
                cv2.rectangle(working_page, (x0, y0), (x1, y1), (255, 0, 0), 2)

        if self.pdf_handler.sub_module_list:
            for bbox in self.pdf_handler.sub_module_list: # x,y,w,h
                x0, y0, x1, y1 = bbox
                cv2.rectangle(working_page, (x0, y0), (x1, y1), (255, 0, 255), 2)

        if self.pdf_handler.steps_area:
            for bbox in self.pdf_handler.steps_area: # x,y,w,h
                x0, y0, x1, y1 = bbox
                cv2.rectangle(working_page, (x0, y0), (x1,y1), (128, 128, 0), 2)

        # Add status text
        status = f"Page {self.pdf_handler.current_page_index + 1}/{self.pdf_handler.pdf_document.page_count}"
        if self.pdf_handler.current_page_index in self.pdf_handler.irrelevant_pages:
            status += " (Irrelevant)"
        cv2.putText(working_page, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(working_page, "Keys: arrows: navigate, i: mark irrelevant, l: select part list, s: select sub-step, p: process, q: quit",
                    (10, working_page.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        # cv2.imshow(self.window_name, working_page)
        show_zoomed_image(self.window_name, working_page)

    #===================================


    def run(self, pdf_name):



        self.pdf_name = pdf_name
        self.pdf_handler = PdfHandler.PdfHandler()
        self.pdf_handler.load_pdf(manuals_folder,pdf_name)


        cv2.namedWindow(self.window_name)
        cv2.setMouseCallback(self.window_name, self.mouse_callback)

        while True:
            self.show_page_decorations()

            key = cv2.waitKey(0) & 0xFF

            # print ("key is",key)

            if key == ord('q'):  # 'q' or Esc to quit
                self.pdf_handler.close(save_meta=True)
                break

            elif key == 27:  # Esc
                self.pdf_handler.close(save_meta=False)
                break

            elif key == 81:  # Left arrow
                self.pdf_handler.prev_image()

            elif key == 83:  # Right arrow
                self.pdf_handler.next_image()

            elif key == ord('i'):  # 'i' to mark irrelevant
                self.pdf_handler.add_page_as_irrelevant()

            elif key == ord('d'):  # 'd' to "do" - process the page
                self.pdf_handler.do_page()

            elif key == ord('p'):  # 'p' for parts processor
                self.pdf_handler.do_parts_page()

            elif key ==  ord('e'):  # 'e' for step extractions
                self.pdf_handler.extract_one_step()


            elif key ==  ord('3'):  # 'e' for step extractions
                self.pdf_handler.extracts_steps()

            elif key ==  ord('z'):  # 'e' for step extractions

                # temp meta dict
                meta_dict = {}
                meta_dict['step_font_size']= 26
                meta_dict['page_sub_step_font_size']= 22
                meta_dict['sub_step_font_size']= 16
                meta_dict['parts_list_font_size']= 8
                meta_dict['page_number_font_size']= 10
                meta_dict['sub_step_color']= [202, 239, 255]
                meta_dict['parts_list_color']= [242, 215, 182]
                # self.parts_list_color = meta_dict.get('parts_list_color', [255,255,255])

                step_processor_v2 = Step_Processor_v3.StepProcessor("processed/"+self.pdf_handler.export_folder,debug_level = 4)
                step_processor_v2.process_doc(self.pdf_handler.pdf_document,2,meta_dict,self.pdf_handler.all_parts_df)

                # page_processor_v2 = Page_Processor_v2.PageProcessor(debug_level=2)
                #
                #
                # #!!!!!!!! here, a full file path should be added to meta. so we know where to put results
                #
                # page_processor_v2.set_meta({})
                # page_processor_v2.process_page(self.pdf_handler.pdf_document,self.pdf_handler.current_page_index)


            elif key in [ord('l'),ord('s'),ord('f'), ord('d')] and self.ref_point:
                # 'l' for list of parts, s for sub step,
                #  f for step font, d for numbered sub step font ?
                (x1, y1), (x2, y2) = self.ref_point
                if key == ord('f'):
                    self.pdf_handler.set_font_rect(self.ref_point,"steps")
                    self.ref_point = []
                    continue

                if key == ord('d'):
                    self.pdf_handler.set_font_rect(self.ref_point,"numbered_substep")
                    self.ref_point = []
                    continue



                current_jpeg = self.pdf_handler.get_img()
                roi = current_jpeg[min(y1, y2):max(y1, y2), min(x1, x2):max(x1, x2)]
                avg_parts_color = np.mean(roi, axis=(0, 1)).astype(np.uint8)
                self.ref_point = []
                if key == ord('l'):
                    print ("parts list color set")
                    self.pdf_handler.set_parts_list_color(avg_parts_color)
                    continue
                if key == ord('s'):
                    print("subset color set")
                    self.pdf_handler.set_sub_step_color(avg_parts_color)
                    continue

        cv2.destroyAllWindows()

        # if self.pdf_document:
        #     self.pdf_document.close()

if __name__ == "__main__":
    parser = LegoInstructionParser()
    # parser.run("Manuals/10698_X_Castle.pdf")
    # parser.run("6186243.pdf") # small buggy
    # parser.run("6217542.pdf")
    # parser.run("6420974.pdf") # 6 wheels "kvadrazikl"
    parser.run("6208467.pdf") # extreme explorer
    # parser.run("4520728.pdf") # old nxt

