import math
import json
import os
import warnings

import cv2
import fitz  # PyMuPDF

import numpy as np

import Step_Processor
import Page_Processor
import Parts_Processor
import exporter
import exporter_by_gpt

top_folder = "processed"


class PdfHandler:
    def __init__(self):
        self.pdf_name = None
        self.export_folder = None

        self.pdf_document = None
        # self.actual_page_count = None
        self.pdf_size = None
        self.current_page_index = 0
        self.current_page_jpeg = None
        self.current_page_pdf = None
        self.irrelevant_pages = set()

        self.meta_loaded = False
        self.meta_dict = {}

        self.steps_numbers = None
        self.parts_list_numbers = None


        self.parts_processor = Parts_Processor.PartsProcessor(debug_level = 0)
        self.all_parts_df = None

        self.step_processor = None
        self.last_page_processor = None
        self.last_steps_block = None

        # self.instruction_step_font_size = None
        # self.numbered_substep_font_size = None
        # self.parts_font_size = None
        #
        # self.parts_list = None
        # self.sub_module_list = None
        # self.steps_area = None



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
            # self.instruction_step_font_size = 26 # default value
            # self.numbered_substep_font_size = 22
            # self.parts_font_size = 8 #default value


            self.meta_loaded = False

            meta_name = pdf_name.replace("pdf", "meta")
            meta_path = os.path.join(self.export_folder, meta_name)
            meta_path = os.path.join(top_folder, meta_path)
            print ("meta path is: ",meta_path)

            # set default meta values
            self.meta_dict['step_font_size'] = 26
            self.meta_dict['numbered_sub_step_font_size'] = 22
            self.meta_dict['sub_step_font_size'] = 16
            self.meta_dict['parts_list_font_size'] = 8
            self.meta_dict['page_number_font_size'] = 10
            self.meta_dict['sub_step_color'] = [202, 239, 255]
            # self.meta_dict['parts_list_color']= [242, 215, 182]
            self.meta_dict['parts_list_color'] = [255, 255, 255]
            self.meta_dict['irrelevant_pages'] = set()

            # override them by file, where possible

            # check if metafile exists
            if os.path.exists(meta_path):
                with open(meta_path, "r") as f:
                    meta_file = json.load(f)
                    if meta_file["step_font_size"] != "n/a":
                        self.meta_dict['step_font_size'] = meta_file["step_font_size"]

                    if meta_file["numbered_sub_step_font_size"] != "n/a":
                        self.meta_dict['numbered_sub_step_font_size'] = meta_file["numbered_sub_step_font_size"]

                    if meta_file["sub_step_font_size"] != "n/a":
                        self.meta_dict['sub_step_font_size'] = meta_file["sub_step_font_size"]

                    if meta_file["parts_list_font_size"] != "n/a":
                        self.meta_dict['parts_list_font_size'] = meta_file["parts_list_font_size"]

                    if meta_file["page_number_font_size"] != "n/a":
                        self.meta_dict['page_number_font_size'] = meta_file["page_number_font_size"]

                    if meta_file["sub_step_color"] != "n/a":
                        self.meta_dict['sub_step_color'] = meta_file["sub_step_color"]

                    if meta_file["parts_list_color"] != "n/a":
                        self.meta_dict['parts_list_color'] = meta_file["parts_list_color"]

                    if meta_file["irrelevant_pages"] != "n/a":
                        self.meta_dict['irrelevant_pages'] = set(meta_file["irrelevant_pages"])
                        self.irrelevant_pages = set(meta_file["irrelevant_pages"])

                    self.meta_loaded = True
                    print ("Meta File Loaded")


            #check if parts list exists
            self.all_parts_df = self.parts_processor.load_parts_list(ex_folder)

            # we don't create page processor, because it's created uniquely for each page

            self.step_processor = Step_Processor.StepProcessor(ex_folder,debug_level = 0)
            self.step_processor.init_doc(self.pdf_document, self.meta_dict,self.all_parts_df)

            # reset variables
            self.current_page_index = -1
            # self.actual_page_count = self.pdf_document.page_count

            print(f"Loaded: {os.path.basename(file_path)}")
        else:
            print("File not found!")
            exit(1)


    def next_image(self):
        tmp = self.current_page_index + 1
        if tmp == self.pdf_document.page_count:
            tmp = 0

        if tmp < self.pdf_document.page_count:
            self.current_page_index = tmp

            '''
            theoretically, only the image should be displayed.
            if it is interesting, a user can call "1" for page processor, and "3" for step processor,
            but i think it will be verry annoying to do it manually each time, so ?
            
            call page processor, pass it's result to step processor.
            if it's ok - we can continue. if not, we can namualy call page processor again, change some meta, or drop images
            then call step processor, and check the results. 
            
            
            !!!! note the problem. page process take a few long second to process. a little bit annoying
            especially for testing.
            
            maybe we can set processing mode for the whole file ?
            the modes may be
            1) only page
            2) collect images
            3) collect steps
            '''

            # do some image loadings
            self.last_page_processor = Page_Processor.PageProcessor()
            self.last_page_processor.set_meta(self.meta_dict)
            self.last_page_processor.prepare_page(self.pdf_document,self.current_page_index)

            # apply some logic
            if self.current_page_index not in self.irrelevant_pages:
                self.last_page_processor.process_page(self.pdf_document,self.current_page_index)
                self.last_steps_block = self.step_processor.process_page(self.last_page_processor)

            #self.step_processor.load_page(self.current_page_index)


    def prev_image(self):
        tmp = self.current_page_index - 1
        if tmp == -1:
            tmp = self.pdf_document.page_count - 1
        if tmp >= 0:
            self.current_page_index = tmp

            self.last_page_processor = Page_Processor.PageProcessor()
            self.last_page_processor.set_meta(self.meta_dict)
            self.last_page_processor.prepare_page(self.pdf_document,self.current_page_index)

            if self.current_page_index not in self.irrelevant_pages:
                self.last_page_processor.process_page(self.pdf_document, self.current_page_index)
                self.last_steps_block = self.step_processor.process_page(self.last_page_processor)

            # self.step_processor.load_page(self.current_page_index)

    def add_page_as_irrelevant(self):
        self.irrelevant_pages.add(self.current_page_index)
        self.meta_dict['irrelevant_pages'] = list (self.irrelevant_pages)
        # self.actual_page_count -= 1

    def set_font_rect(self,ref_point,font_class):
        (x1, y1), (x2, y2) = ref_point
        xc = x1 + (x2-x1)/2
        yc = y1 + (y2-y1)/2
        res = self.__get_step_font_size__(xc,yc)
        if res:
            if font_class == "steps":
                self.meta_dict['step_font_size'] = res
            elif font_class == "numbered_substep":
                self.meta_dict['numbered_sub_step_font_size'] = res
            elif font_class == "sub_step_font_size":
                self.meta_dict['sub_step_font_size'] = res
            elif font_class == "parts":
                self.meta_dict['parts_list_font_size'] = res

            # self.step_processor = Step_Processor.StepProcessor(self)
            print (font_class + " font size set to: ",res)
            self.step_processor.pp.set_meta(self.meta_dict)
        else:
            warnings.warn("failed font selection")

        '''
        here we got a rect with some step num. we have to find the closest rect
        so actualy we need a center of this rect, and centers of all other
        then find a dist to each one, and chose the minimal
        '''

    def set_parts_list_color(self, color):
        self.meta_dict['parts_list_color'] = color
        self.last_page_processor.set_meta(self.meta_dict)

    def set_sub_step_color(self, color):
        self.meta_dict['sub_step_color'] = color
        self.last_page_processor.set_meta(self.meta_dict)

    def do_parts_page(self):
        # don't forget top folder
        if self.parts_processor:
            target_folder = os.path.join(top_folder, self.export_folder)
            self.all_parts_df =  self.parts_processor.extract_parts(target_folder, self.pdf_document, self.current_page_index)
        else:
            warnings.warn("parts processor not defined")


    def close(self, save_meta):

        # here I want to save meta file
        #if we have at least step font size, and parts list - we are good to go
        if save_meta:

            meta = {}
            meta['step_font_size'] = self.meta_dict['step_font_size']
            meta['numbered_sub_step_font_size'] = self.meta_dict['numbered_sub_step_font_size']
            meta['sub_step_font_size'] = self.meta_dict['sub_step_font_size']
            meta['parts_list_font_size'] = self.meta_dict['parts_list_font_size']
            meta['page_number_font_size'] = self.meta_dict['page_number_font_size']

            meta["sub_step_color"] = self.meta_dict['sub_step_color']
            meta["parts_list_color"] = self.meta_dict['parts_list_color']
            meta["irrelevant_pages"] = list(self.meta_dict['irrelevant_pages'])

            meta_name = self.pdf_name.replace("pdf", "meta")
            meta_path = os.path.join(self.export_folder, meta_name)
            meta_path = os.path.join(top_folder, meta_path)
            # print ("metadata path: ",meta_path)
            # print("metadata data: ", meta)

            with open(meta_path, "w") as f:
                json.dump(meta, f, indent=4)

            # save the first page, for reference of what model is build
            self.last_page_processor = Page_Processor.PageProcessor()
            self.last_page_processor.set_meta(self.meta_dict)
            self.last_page_processor.prepare_page(self.pdf_document, 0)
            image = self.last_page_processor.page_image.copy()
            output_image_path = os.path.join(self.export_folder, "first_page.jpg")
            output_image_path = os.path.join(top_folder, output_image_path)
            # print ("img path ",output_image_path)
            cv2.imwrite(output_image_path, image)

        if self.pdf_document:
            self.pdf_document.close()


    #############################

    def reload_page_processor(self):
        '''
        here i want to reload only page processor
        so "show page decorations" can be called
        '''
        self.last_page_processor = Page_Processor.PageProcessor()
        self.last_page_processor.set_meta(self.meta_dict)
        self.last_page_processor.prepare_page(self.pdf_document, self.current_page_index)

    def redo_step_processor(self):
        self.last_page_processor.process_page(self.pdf_document, self.current_page_index)
        self.last_steps_block = self.step_processor.process_page(self.last_page_processor)

    def prepare_page(self):
        return self.step_processor.prepare_page(self.current_page_index)

    def process_page(self,page_processor):
        return self.step_processor.process_page(page_processor)

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
        # print("sub modules:", len(self.sub_module_list))
        #
        # target_folder = os.path.join(top_folder, self.export_folder)
        # self.step_processor.extract_step(target_folder, self.pdf_document, self.current_page_index,
        #                                  self.steps_area, self.all_parts_df,self.sub_module_list)
        b = self.step_processor.last_blocks[0]
        # exporter.save_step_block(b,"./test_output")
        exporter_by_gpt.save_step_block(b,"./test_output_gpt")


    # def extracts_steps(self):
    #     '''
    #     here I expect to get all steps from one page.
    #     theoreticly, i have to get an object, contains all step element, because i've already found them in detect_Steps_area
    #
    #     '''
    #     if self.step_processor:
    #
    #         # print (self.all_parts_list[5])
    #
    #
    #         target_folder = os.path.join(top_folder, self.export_folder)
    #         # self.sp.extract_step(target_folder,self.pdf_document, self.current_page_index,self.steps_area)
    #         for i in range (1,self.pdf_document.page_count):
    #             if i not in self.irrelevant_pages:
    #                 print ("extracting pdf page: ",i)
    #                 self.current_page_index = i
    #                 self.__update_page_image__()
    #                 self.__extract_pdf_page__()
    #                 self.__process_page__()
    #
    #                 print ("sub modules:", len(self.sub_module_list))
    #
    #                 self.step_processor.extract_step(target_folder, self.pdf_document, i, self.steps_area, self.all_parts_df,None)
    #     else:
    #         warnings.warn("failed to run step extractor. font not set ?")

    def get_img(self):
        if self.last_page_processor.page_image is not None:
            return self.last_page_processor.page_image.copy()
        else:
            return None

    def get_last_blocks(self):
        return self.last_steps_block

    #############################################
    ###
    ###   some logic here, till we build a separate class for it

    def __get_step_font_size__(self,x,y):

        # get texts from PDF


        page = self.pdf_document[self.current_page_index]


        for block in page.get_text("dict")["blocks"]:
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