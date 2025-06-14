import math
import json
import os
import warnings

import fitz  # PyMuPDF
import cv2
import numpy as np

import Page_Processor
import Parts_Processor
import Step_Processor

top_folder = "processed"


class PdfHandler:
    def __init__(self):
        self.pdf_name = None
        self.export_folder = None

        self.pdf_document = None
        self.actual_page_count = None
        self.pdf_size = None
        self.current_page_index = 0
        self.current_page_jpeg = None
        self.current_page_pdf = None
        self.irrelevant_pages = set()

        self.meta_loaded = False

        self.steps_numbers = None
        self.partslist_numbers = None

        self.page_processor = Page_Processor.PageProcessor(debug = False)
        self.parts_processor = Parts_Processor.PartsProcessor(debug = False)
        self.all_parts_df = None

        self.step_processor = None
        self.instruction_step_font_size = None
        self.numbered_substep_font_size = None
        self.parts_font_size = None

        self.parts_list = None
        self.sub_module_list = None
        self.steps_area = None



    def load_pdf(self, manuals_folder, pdf_name):
        file_path = os.path.join(manuals_folder, pdf_name)

        if os.path.exists(file_path):
            # load file
            self.pdf_document = fitz.open(file_path)
            self.pdf_name = pdf_name

            self.export_folder = pdf_name.replace(".pdf", "")
            ex_folder = os.path.join(top_folder, self.export_folder)
            os.makedirs(ex_folder, exist_ok=True)

            self.irrelevant_pages.clear()
            self.instruction_step_font_size = 26 # default value
            self.numbered_substep_font_size = 22
            self.parts_font_size = 8 #default value
            self.step_processor = Step_Processor.StepProcessor(self)
            self.meta_loaded = False

            meta_name = pdf_name.replace("pdf", "meta")
            meta_path = os.path.join(self.export_folder, meta_name)
            meta_path = os.path.join(top_folder, meta_path)
            print ("meta path is: ",meta_path)

            # check if metafile exists
            if os.path.exists(meta_path):
                with open(meta_path, "r") as f:
                    meta = json.load(f)

                    self.instruction_step_font_size = meta["step_font_size"]

                    self.step_processor = Step_Processor.StepProcessor(self)

                    self.page_processor.set_parts_list_color(np.array(meta["parts_list_color"]))

                    if meta["numbered_substep_font_size"] != "n/a":
                        self.numbered_substep_font_size = meta["numbered_substep_font_size"]

                    if meta["sub_step_color"] != "n/a":
                        self.page_processor.set_sub_step_color(np.array(meta["sub_step_color"]))

                    if meta["irrelevant_pages"] != "n/a":
                        self.irrelevant_pages = set(meta["irrelevant_pages"])

                    self.meta_loaded = True

            #check if parts list exists
            self.all_parts_df = self.parts_processor.load_parts_list(ex_folder)


            # reset variables

            self.current_page_index = 0

            self.actual_page_count = self.pdf_document.page_count

            # update page size
            self.current_page_pdf = self.pdf_document[self.current_page_index]
            self.pdf_size = (int(self.current_page_pdf.rect.width), int(self.current_page_pdf.rect.height))

            print(f"Loaded: {os.path.basename(file_path)}")
            self.__update_page_image__()
        else:
            print("File not found!")
            exit(1)

    def __update_page_image__(self):
        if not self.pdf_document or self.current_page_index >= self.actual_page_count:
            return None
        self.current_page_pdf = self.pdf_document[self.current_page_index]
        pix = self.current_page_pdf.get_pixmap(matrix=fitz.Matrix(1, 1))
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3)
        # Create a writable copy to avoid read-only error
        img = img.copy()
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
        self.current_page_jpeg = img

        # clean decorations
        self.steps_numbers = []
        self.partslist_numbers = []
        self.parts_list = []
        self.sub_module_list = []
        self.steps_area = []

        return None

    def next_image(self):
        tmp = self.current_page_index + 1
        while tmp in self.irrelevant_pages:
            tmp += 1
            # allow circular paging
            if tmp == self.actual_page_count:
                tmp = 0

        if tmp < self.actual_page_count:
            self.current_page_index = tmp
            self.__update_page_image__()
            if self.meta_loaded:
                self.do_page()

    def prev_image(self):
        tmp = self.current_page_index - 1
        while tmp in self.irrelevant_pages:
            tmp -= 1
            # allow circular paging
            if tmp == -1:
                tmp = self.actual_page_count-1


        if tmp >= 0:
            self.current_page_index = tmp
            self.__update_page_image__()
            if self.meta_loaded:
                self.do_page()

    def add_page_as_irrelevant(self):
        self.irrelevant_pages.add(self.current_page_index)
        # self.actual_page_count -= 1

    def set_font_rect(self,ref_point,font_class):
        (x1, y1), (x2, y2) = ref_point
        xc = x1 + (x2-x1)/2
        yc = y1 + (y2-y1)/2
        res = self.__get_step_font_size__(xc,yc)
        if res:
            if font_class == "steps":
                self.instruction_step_font_size = res
            elif font_class == "numbered_substep":
                self.numbered_substep_font_size = res
            # elif font_class == "rounded_substep":
            #     self.numbered_substep_font_size = res
            elif font_class == "parts":
                self.parts_font_size = res

            self.step_processor = Step_Processor.StepProcessor(self)
            print (font_class + " font size set to: ",res)
        else:
            warnings.warn("failed font selection")

        '''
        here we got a rect with some step num. we have to find the closest rect
        so actualy we need a center of this rect, and centers of all other
        then find a dist to each one, and chose the minimal
        '''

    def set_parts_list_color(self, color):
        self.page_processor.set_parts_list_color(color)

    def set_sub_step_color(self, color):
        self.page_processor.set_sub_step_color(color)

    def do_parts_page(self):
        # don't forget top folder
        if self.parts_processor:
            target_folder = os.path.join(top_folder, self.export_folder)
            self.all_parts_df =  self.parts_processor.extract_parts(target_folder, self.pdf_document, self.current_page_index)
        else:
            warnings.warn("parts processor not defined")


    def do_page(self):
        self.__extract_pdf_page__()
        self.__process_page__()

    def close(self, save_meta):
        if self.pdf_document:
            self.pdf_document.close()

        # here I want to save meta file
        #if we have at least step font size, and parts list - we are good to go
        if save_meta:
            if self.instruction_step_font_size and self.page_processor.parts_list_color is not None:
                meta = {}
                meta["step_font_size"] = self.instruction_step_font_size
                meta["parts_list_color"] = self.page_processor.parts_list_color.tolist()
                meta["numbered_substep_font_size"] = self.numbered_substep_font_size


                if self.page_processor.sub_step_color is not None:
                    meta["sub_step_color"] = self.page_processor.sub_step_color.tolist()
                else:
                    meta["sub_step_color"] = "n/a"

                if len(self.irrelevant_pages) != 0:
                    meta["irrelevant_pages"] = list(self.irrelevant_pages)
                else:
                    meta["irrelevant_pages"] = "n/a"

                meta_name = self.pdf_name.replace("pdf", "meta")
                meta_path = os.path.join(self.export_folder, meta_name)
                meta_path = os.path.join(top_folder, meta_path)
                print ("metadata: ",meta_path)

                with open(meta_path, "w") as f:
                    json.dump(meta, f, indent=4)


    def __extract_pdf_page__(self):
        # self.bbox_list = []

        # get steps numbers
        steps = []
        list_numbers = []
        numbered_substeps_number = []

        # get texts from PDF
        for block in self.current_page_pdf.get_text("dict")["blocks"]:
            if "lines" in block:
                for line in block["lines"]:
                    for span in line["spans"]:
                        text = span["text"]
                        pos = [int(v) for v in span["bbox"]]
                        font_size = span["size"]

                        # get step numbers by size
                        if self.instruction_step_font_size and font_size == self.instruction_step_font_size:
                            steps.append((text, pos))

                        if self.numbered_substep_font_size and font_size == self.numbered_substep_font_size:
                            numbered_substeps_number.append((text,pos))

                        # get parts list numbers by size
                        if font_size == 8 and 'x' in text:  # means it's parts font
                            list_numbers.append((text, pos))

                        # print (f"Text: {text}\n  Position: {pos}\n  Font size: {font_size}\n")

        self.steps_numbers = sorted(steps, key=lambda x: int(x[0]))
        self.partslist_numbers = list_numbers  # sorted(list_numbers, key=lambda x: int(x[0]))

    def __process_page__(self):

        self.parts_list = self.page_processor.detect_parts_list(self.current_page_jpeg, self.partslist_numbers)
        self.sub_module_list = self.page_processor.detect_sub_steps(self.current_page_jpeg)

        self.steps_area = self.page_processor.detect_steps_area(self.steps_numbers, self.parts_list, self.pdf_size)

    def extract_one_step(self):
        print("sub modules:", len(self.sub_module_list))

        target_folder = os.path.join(top_folder, self.export_folder)
        self.step_processor.extract_step(target_folder, self.pdf_document, self.current_page_index,
                                         self.steps_area, self.all_parts_df,self.sub_module_list)


    def extracts_steps(self):
        '''
        here I expect to get all steps from one page.
        theoreticly, i have to get an object, contains all step element, because i've already found them in detect_Steps_area

        '''
        if self.step_processor:

            # print (self.all_parts_list[5])


            target_folder = os.path.join(top_folder, self.export_folder)
            # self.sp.extract_step(target_folder,self.pdf_document, self.current_page_index,self.steps_area)
            for i in range (1,self.pdf_document.page_count):
                if i not in self.irrelevant_pages:
                    print ("extracting pdf page: ",i)
                    self.current_page_index = i
                    self.__update_page_image__()
                    self.__extract_pdf_page__()
                    self.__process_page__()

                    print ("sub modules:", len(self.sub_module_list))

                    self.step_processor.extract_step(target_folder, self.pdf_document, i, self.steps_area, self.all_parts_df,None)
        else:
            warnings.warn("failed to run step extractor. font not set ?")

    def get_img(self):
        # return self.current_page_jpeg
        return self.current_page_jpeg.copy()

    #############################################
    ###
    ###   some logic here, till we build a separate class for it

    def __get_step_font_size__(self,x,y):

        # get texts from PDF
        for block in self.current_page_pdf.get_text("dict")["blocks"]:
            if "lines" in block:
                for line in block["lines"]:
                    for span in line["spans"]:
                        text = span["text"]
                        pos = [int(v) for v in span["bbox"]]
                        font_size = span["size"]

                        x1,y1,x2,y2 = pos

                        if x1 < x < x2 and y1 < y < y2:
                            # print ("click in ",text)
                            return font_size
        return None