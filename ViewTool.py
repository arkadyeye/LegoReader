import fitz  # PyMuPDF
import cv2
import numpy as np
import os

import PageProcessor

zoom = 1.5
def show_zoomed_image(window_name, cv_image):
    height, width = cv_image.shape[:2]
    new_size = (int(width * zoom), int(height * zoom))
    resized = cv2.resize(cv_image, new_size, interpolation=cv2.INTER_CUBIC)
    cv2.imshow(window_name, resized)


class LegoInstructionParser:
    def __init__(self):
        self.window_name = "LEGO Instruction Parser"

        self.pdf_document = None
        self.pdf_size = None
        self.current_page_index = 0
        self.current_page_jpeg = None

        self.pp = PageProcessor.PageProcessor(False)
        self.irrelevant_pages = set()

        self.selection_start = None
        self.selection_end = None
        self.drawing = False

        self.instruction_step_font_size = 0
        self.bbox_list = []

        # vars for rectangle color selection
        self.ref_point = []
        self.cropping = False
        self.image = None
        self.clone = None
        self.preview_image = None

        #vars for displaying a frame around elements
        self.parts_list = []
        self.sub_module_list = []
        self.steps_number_list = []
        self.parts_number_list = []
        self.steps_area = []

    def load_pdf(self, file_path):
        if os.path.exists(file_path):
            self.pdf_document = fitz.open(file_path)
            self.current_page_index = 0
            self.irrelevant_pages.clear()

            self.instruction_step_font_size = 26.0 # some pdfs has 30. make setting file !!!!!

            print(f"Loaded: {os.path.basename(file_path)}")
            self.update_page_image()
        else:
            print("File not found!")
            exit(1)

    def update_page_image(self):
        if not self.pdf_document or self.current_page_index >= self.pdf_document.page_count:
            return None
        page = self.pdf_document[self.current_page_index]
        pix = page.get_pixmap(matrix=fitz.Matrix(1, 1))  # Zoom for better resolution
        #pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))  # Zoom for better resolution
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3)
        # Create a writable copy to avoid read-only error
        img = img.copy()
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
        self.current_page_jpeg = img

        # reset page decorations
        self.steps_number_list = []
        self.parts_list = []
        self.sub_module_list = []
        self.parts_number_list = []
        self.steps_area = []

        return None

    def mouse_callback(self,event, x, y, flags, param):

        # anti-zoom mouse coordinates
        x = int(x/zoom)
        y = int(y/zoom)

        if event == cv2.EVENT_LBUTTONDOWN:
            self.ref_point = [(x, y)]
            self.cropping = True
            self.preview_image = self.current_page_jpeg.copy()

        elif event == cv2.EVENT_MOUSEMOVE:
            if self.cropping:
                self.preview_image = self.current_page_jpeg.copy()
                cv2.rectangle(self.preview_image, self.ref_point[0], (x, y), (0, 255, 0), 2)

                # cv2.imshow(self.window_name, self.preview_image)
                show_zoomed_image(self.window_name, self.preview_image)

        elif event == cv2.EVENT_LBUTTONUP:
            self.ref_point.append((x, y))
            self.cropping = False
            self.preview_image = self.current_page_jpeg.copy()
            cv2.rectangle(self.preview_image, self.ref_point[0], self.ref_point[1], (0, 255, 0), 2)
            # cv2.imshow(self.window_name, self.preview_image)
            show_zoomed_image(self.window_name, self.preview_image)

    def show_page_decorations(self):
        # self.update_page_image()
        if self.current_page_jpeg is None:
            return
        working_page = self.current_page_jpeg.copy()

        # surround steps numbers
        if self.steps_number_list:
            for num,bbox in self.steps_number_list:
                # x0, y0, x1, y1 = [int(v * self.zoom) for v in bbox] # Adjust for zoom
                x0, y0, x1, y1 = [int(v) for v in bbox] # Adjust for zoom

                cv2.rectangle(working_page, (x0, y0), (x1, y1), (0, 255, 0), 2)
                # cv2.putText(img, str(i),  (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        if  self.parts_number_list:
            for num,bbox in  self.parts_number_list:
                # x0, y0, x1, y1 = [int(v * self.zoom) for v in bbox] # Adjust for zoom
                x0, y0, x1, y1 = [int(v) for v in bbox] # Adjust for zoom
                cv2.rectangle(working_page, (x0, y0), (x1, y1), (0, 255, 0), 2)
                # cv2.putText(img, str(i),  (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)




        if self.parts_list:
            for bbox in self.parts_list: # x,y,w,h
                x0, y0, x1, y1 = bbox # no need for zoom, as already zoomed image passed to processor
                cv2.rectangle(working_page, (x0, y0), (x1, y1), (255, 0, 0), 2)
                # cv2.putText(img, str(i),  (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        if self.sub_module_list:
            for bbox in self.sub_module_list: # x,y,w,h
                x0, y0, x1, y1 = bbox # no need for zoom, as already zoomed image passed to processor
                cv2.rectangle(working_page, (x0, y0), (x1, y1), (255, 0, 255), 2)
                # cv2.putText(img, str(i),  (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)


        if self.steps_area:
            for bbox in self.steps_area: # x,y,w,h
                x0, y0, x1, y1 = [int(v) for v in bbox]
                cv2.rectangle(working_page, (x0, y0), (x1,y1), (128, 128, 0), 2)



        # Draw part list rectangle if selected
        i = 0
        if self.bbox_list:
            for bbox in self.bbox_list:
                # print ("box",bbox)
                #x0, y0, x1, y1 = [int(v * self.zoom) for v in bbox]
                x0, y0, x1, y1 = [int(v) for v in bbox]
                # x0, y0, x1, y1 = [c * 2 for c in box]  # Adjust for zoom
                cv2.rectangle(self.current_page_jpeg, (x0, y0), (x1, y1), (0, 255, 0), 2)
                # cv2.putText(img, str(i),  (x1+10*i, y1), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                i += 1
        # # Draw selection rectangle while dragging
        # if self.drawing and self.selection_start and self.selection_end:
        #     cv2.rectangle(img, self.selection_start, self.selection_end, (255, 0, 0), 2)

        # Add status text
        status = f"Page {self.current_page_index + 1}/{self.pdf_document.page_count}"
        if self.current_page_index in self.irrelevant_pages:
            status += " (Irrelevant)"
        cv2.putText(working_page, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(working_page, "Keys: arrows: navigate, i: mark irrelevant, l: select part list, s: select sub-step, p: process, q: quit",
                    (10, working_page.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        # cv2.imshow(self.window_name, working_page)
        show_zoomed_image(self.window_name, working_page)

    #===================================

    def process_page(self,current_page):

        self.bbox_list = []

        # here I want to get data from pdf doc, and mark each element with it's href
        page = self.pdf_document[current_page]

        self.pdf_size = (page.rect.width, page.rect.height)
        # print ("pdf size: "+str(size))

        # get steps numbers
        steps = []
        list_numbers = []


        for block in page.get_text("dict")["blocks"]:
            if "lines" in block:
                for line in block["lines"]:
                    for span in line["spans"]:
                        text = span["text"]
                        # pos = span["bbox"]  # (x0, y0, x1, y1)
                        pos = [int(v) for v in span["bbox"]]
                        font_size = span["size"]

                        if font_size == self.instruction_step_font_size:
                            steps.append((text,pos))

                        if font_size == 8 and 'x' in text: #means it's parts font
                            list_numbers.append((text,pos))

                        #print (f"Text: {text}\n  Position: {pos}\n  Font size: {font_size}\n")

        self.steps_number_list = sorted(steps, key=lambda x: int(x[0]))
        # print ("steps sorted_list: ",self.steps_number_list)

        self.parts_number_list = list_numbers#sorted(list_numbers, key=lambda x: int(x[0]))
        # print("parts sorted_list: ", self.parts_number_list)

        # get parts list bounding box
        self.parts_list = self.pp.detect_parts_list(self.current_page_jpeg,list_numbers)
        # print("parts_list: ", self.parts_list)

        self.sub_module_list = self.pp.detect_sub_steps(self.current_page_jpeg)
        # print("parts_list: ", self.sub_module_list)

        self.steps_area = self.pp.detect_steps_area(self.steps_number_list,self.parts_list,self.pdf_size)

        # self.parts_list = []
        # self.sub_module_list = []
        # self.steps_number_list = []




    def run(self, pdf_path):
        self.load_pdf(pdf_path)
        cv2.namedWindow(self.window_name)
        cv2.setMouseCallback(self.window_name, self.mouse_callback)

        while True:
            self.show_page_decorations()
            # print("in while")
            #
            # тут надо как то решить, чтобы он постоянно обновлял, толко
            # когда я кроп делаю. можно кроп мод включать кнопокй.
            # а кнопкой принятия решения выключать.
            # и тогда с превию имаге проще :)
            #
            # кнопка включающея "кроп мод" - их две (точнее три)
            # одна включает выбор "парт лист"
            # вторая включает выбор "саб таск"
            # третия включает выбор фотна - там по началу тоже кроп, но в итоге вызывается функция про пдф

            key = cv2.waitKey(0) & 0xFF
            #
            # if self.cropping:
            #     key = cv2.waitKey(1) & 0xFF
            # else:
            #     key = cv2.waitKey(0) & 0xFF

            if key in [ord('q'), 27]:  # 'q' or Esc to quit
                break

            elif key == 81:  # Left arrow
                if self.current_page_index > 0:
                    self.current_page_index -= 1
                    self.update_page_image()
            elif key == 83:  # Right arrow
                if self.current_page_index < self.pdf_document.page_count - 1:
                    self.current_page_index += 1
                    self.update_page_image()

            elif key == ord('d'):  # 'i' to mark irrelevant
                self.irrelevant_pages.add(self.current_page_index)
            elif key == ord('p'):  # 'p' to process the page
                self.process_page(self.current_page_index)
            elif key in [ord('l'),ord('s')] and self.ref_point:  # 'l' for list of parts
                (x1, y1), (x2, y2) = self.ref_point
                roi = self.current_page_jpeg[min(y1, y2):max(y1, y2), min(x1, x2):max(x1, x2)]
                avg_parts_color = np.mean(roi, axis=(0, 1)).astype(np.uint8)
                self.ref_point = []
                if key == ord('l'):
                    print ("parts list color set")
                    self.pp.set_parts_list_color(avg_parts_color)
                if key == ord('s'):
                    print("subset color set")
                    self.pp.set_sub_step_color(avg_parts_color)


                # parts_rects = pp.detect_parts_list(clone)
                # sub_rects = pp.detect_sub_steps(clone)

            # elif key == ord('i'):  # 'i' to mark irrelevant
            #     self.irrelevant_pages.add(self.current_page)
            #     print(f"Marked page {self.current_page + 1} as irrelevant")
            # elif key == ord('s'):  # 's' to select part list
            #     print("Click and drag to select part list region, release to confirm")
            #     # Selection handled by mouse callback
            # elif key == ord('p'):  # 'p' to process
            #     self.process_pdf()

        cv2.destroyAllWindows()
        if self.pdf_document:
            self.pdf_document.close()

if __name__ == "__main__":
    parser = LegoInstructionParser()
    # parser.run("10698_X_Castle.pdf")
    # parser.run("6186243.pdf")
    # parser.run("6217542.pdf")
    parser.run("Manuals/6420974.pdf")