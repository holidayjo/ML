import os
import pandas as pd
import glob
import shutil
import csv
import numpy as np
import cv2
from multiprocessing import Pool
from bs4 import BeautifulSoup

def crop_images(img_path, output_path, h1,h2,w1,w2): # ratio = h1:h2, w1:w2   , add "start_point, end_point," next time 
    
    all_img_files = glob.glob(img_path + '/*.jpg')
    # print(all_img_files)
    for img in all_img_files:
        print(img)
        crop_1image(img, output_path, h1,h2,w1,w2)
    print('Finished!!!')

def crop_1image(image_path, out_path, *ratio):
    '''
    crop ratio is the ratio from the top and left of the image to be removed from the orig img.
    '''
    
    img      = cv2.imread(image_path)
    # i_w = img.shape[1]
    # i_h = img.shape[0]
    
    base_name = os.path.basename(image_path)
    # print(crop_ratio)
    # crop_img = img[h1:h2,w1:w1]

def crop_1image_320(image_path, out_path, *crop_ratio):
    '''
    crop ratio is the ratio from the top and left of the image to be removed from the orig img.
    '''
    
    img      = cv2.imread(image_path)
    i_w = img.shape[1]
    i_h = img.shape[0]
    
    base_name = os.path.basename(image_path)
    
    crop_img = img[int(i_h*crop_ratio):i_h, int(i_w*crop_ratio):i_w]
    # cv2.imshow('cropped', crop_img)
    # cv2.waitKey(0)
    cropped_file = os.path.join(out_path, 'crop_{}'.format(base_name))
    # print(cropped_file)
    cv2.imwrite(cropped_file, crop_img)    



def fps_check(video_file):
    '''
    video_file: ex) r'/media/hj/Docs/my_doc/safety_2022/videos/dongjak/dongjak4_221019_17_20/dongjak4_221019_17_20.mp4'
    '''
    cap          = cv2.VideoCapture(video_file)
    # print(video_file)
    print(round(cap.get(cv2.CAP_PROP_FPS)))
    


def time_to_seconds(time_str):
    h, m, s = map(int, time_str.split(':'))
    total_seconds = h * 3600 + m * 60 + s
    return total_seconds


def xml_checker_n_fixer(xml_file):
    '''
    Some xml files are encoded in a different way. 
    This function repairs these files and returns encoded xml file data.
    '''
    # with open(xml_file, 'r', encoding='UTF8') as f:
    with open(xml_file, 'r') as f:
        first_line = next(f)
        print('first line =', first_line)
        f.seek(0)
        if 'ï»¿' in first_line:
            # print("The first line includes ï»¿. --> encoding='utf-8-sig'")
            with open(xml_file, 'r', encoding="utf-8-sig") as f:
                data = f.read()
            # print('data =', data)
            return data
        else:
            # print("The first line does not include ï»¿ --> normal encoding.")
            with open(xml_file, 'r') as f:
            # with open(xml_file, 'r', encoding='UTF8') as f:
                data = f.read()
                # print('data =', data)
            return data

def start_time_from_xml(xml_file):
    xml_data = xml_checker_n_fixer(xml_file)
    # xml_data = xml_checker_n_fixer(xml_data)
    # print('xml_data =', xml_data)
    soup     = BeautifulSoup(xml_data, 'xml')
    print(soup)
    
    start_time = soup.find('StartTime').text
    print("StartTime:", start_time)
    print('sec =', time_to_seconds(start_time))
    
    return time_to_seconds(start_time)
    

def extract_frames_1video_period(video_path, output_dir, start_time, end_time, frame_interv=30):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.set(cv2.CAP_PROP_POS_MSEC, start_time*1000) # set the video to the start time. in [ms]
    ffile_name    = os.path.basename(video_path)
    file_name     = os.path.splitext(ffile_name)[0]
    end_frame_num = fps*end_time
    # print(fps)
    while cap.isOpened():
        ret, frame = cap.read()
        frame_num = int(cap.get(1))
        # print(frame_num)
        
        if not ret:
            print('Finished!!!!')
            break
        
        if frame_num % frame_interv == 0:
            print(ffile_name,':', frame_num, '[frame]')
            cv2.imwrite(os.path.join(output_dir, '%s_%06d.jpg' % (file_name, int(cap.get(1)))), frame)
        
        if frame_num > end_frame_num:
            break
    cap.release()


def extract_frames_1video(input_path, frame_interv, out_dir):
    '''
    input path is the full path of the video file.             ex) r'/media/hj/Docs/my_doc/videos/gangnam/splitted/gangnam_221011_000.avi'
    out_dir is directory that extracted frames will be saved.  ex) r'/media/hj/Docs/my_doc/videos/gangnam/splitted/frames'
    '''
    
    path         = input_path

    output_path  = out_dir # os.path.join(os.path.dirname(path), 'frames')
    print('output_path =', output_path)
    ffile_name   = os.path.basename(path)
    cap          = cv2.VideoCapture(path)
    
    # frame_width  = int(cap.get(3))
    # frame_height = int(cap.get(4))
    # fps          = round(cap.get(cv2.CAP_PROP_FPS)) # current frame = cap.get(1)
    
    # os.chdir(dir)
    
    file_name    = os.path.splitext(ffile_name)[0]
    
    # print(os.path.join(out_dir, '%s_%06d.jpg' % (file_name, fps)))

    while cap.isOpened():
        ret, frame = cap.read()
        frame_num  = int(cap.get(1))
        
        if not ret:
            print('Finished!!!!')
            break
        
        if frame_num % frame_interv == 0:
            print(ffile_name,':', frame_num, '[frame]')
            cv2.imwrite(os.path.join(output_path, '%s_%06d.jpg' % (file_name, int(cap.get(1)))), frame)
        
    
        # cv2.imshow('frame', frame)
        # if cv2.waitKey(1) == ord('q'):
        #     break
    cap.release()
    # cv2.destroyAllWindows()

    

def counting_class_in_1file(label_file, num_of_classes):
    '''
    Counting class in 1 txt label file.
    We need to input how many classes there are.    
    '''
    
    classes   = np.arange(num_of_classes, dtype=int)
    num_class = np.zeros(num_of_classes, dtype=int)
    
    line_num = 0
    
    # opening file.
    with open(label_file, 'r') as file4line:
        
        # reding each line    
        for line in file4line:
            # print('line =', line)
            line_num += 1
            
            word_num = 0
            
            # readng 1st word (class in yolo format)
            for word in line.split():
                word_num += 1

                if word_num != 1:
                    break
                           
                # looping each class
                for cls_num in classes:
                    # print('cls_num =', cls_num)
                    if word == str(classes[cls_num]):
                        num_class[cls_num] += 1
    # print('num_class =', num_class)
            
    return num_class

            



def moving_only_male_files(img_path, moving_path, *label_path, num_of_classes=2):
    '''
    I found that there are more male than female in the station CCTV.
    We need to balance the dataset. So I decided to remove frames that have only male.
    This function moves images and labels which have only male.
    
    img_path = label_path
    '''
    if label_path == ():
        label_path = img_path
        
    print('From: ', label_path)
    print('To  : ', moving_path)
        
    all_txt_files = glob.glob(label_path + '/*.txt')
    # print(all_txt_files)
    
    # reading each txt file.
    for file in all_txt_files: 
        # print(file)
        num_class = counting_class_in_1file(file, num_of_classes)
        if num_class[-1] == 0:
            # print(os.path.basename(file))
            base_name   = os.path.basename(file)
            splited     = os.path.splitext(base_name)[0]
            moving_file = os.path.join(moving_path, os.path.basename(file))
            
            label_orig = file
            label_move = os.path.join(moving_path, os.path.basename(file))
            
            img_orig   = os.path.join(img_path, splited+'.jpg')
            img_move   = os.path.join(moving_path, splited+'.jpg')
            print(img_orig)
            print(img_move)
            # print(base_name)
            # print(splited)
            shutil.move(label_orig, label_move)
            shutil.move(img_orig,   img_move)
    print('\nFinished!!!!')
    
    
def moving_files_each40frame(orig_img, orig_label, dest_path, frame_select=40):
    '''
    It moves img and labels of each 'frame_selected' frames to the dest folder.
    '''
    
    for f_40s in os.listdir(orig_img):
        frame_num = f_40s[29:33]
        if int(frame_num) % frame_select == 0:
                
            print(f_40s)
            # print(frame_num)
            
            label_name = os.path.splitext(f_40s)
            label_name = label_name[0]+'.txt'
            
            full_orig_image = os.path.join(orig_img, f_40s)
            full_dest_image = os.path.join(dest_path, f_40s)

            full_orig_label = os.path.join(orig_label, label_name)
            full_dest_label = os.path.join(dest_path, label_name)
            
            # print(full_orig_image)
            # print(full_dest_image)
            
            # print(full_orig_label)
            # print(full_dest_label)
            
                        
            
            shutil.copy(full_orig_image, full_dest_image)
            shutil.copy(full_orig_label, full_dest_label)

def delete_classes(input_path, output_path):
    '''
    It deletes all other lines (or classes) detected in the txt file.
    '''
    
    all_txt_files = glob.glob(input_path + '/*.txt')
    for file in all_txt_files: # each file
        # print(file)
        input_file_name = os.path.basename(file)
        
        # print(file_name)
        output_file_name = output_path+'/'+input_file_name
        print('Output file name =', output_file_name)
        
        # reading each line
        with open(file, 'r') as file4line:
            lines     = file4line.readlines()
            
            # print(lines)
            
        with open(output_file_name, 'w') as file_deleted:
            for line in lines:                   # Reading each line.
                # print(line)
                
                # word_num = 1
                for word in line.split():        # Reading each word.
                    
                    if word == str(0):
                        # print(word) # 0
                        
                        file_deleted.write(line)
                    break




    
def image_resize(img_path, out_path, resize_ratio=0.5):
    all_img_files = glob.glob(img_path + '/*.jpg')
    all_files = len(all_img_files)
    print(all_files)
    for i, img in enumerate(all_img_files):
        print('{}/{}'.format(i, all_files))
        
        each_img = cv2.imread(img)
        scale = (int(each_img.shape[1] * resize_ratio), int(each_img.shape[0] * resize_ratio))
        resized  = cv2.resize(each_img, scale, interpolation = cv2.INTER_AREA)
        cv2.imwrite(os.path.join(out_path, os.path.basename(img)), resized)
        # print(os.path.basename(img))
        # img_from = img
        # img_to   = os.path.join(out_path, os.path.basename(img))
        # print(img_from)
        # print(img_to)
        # shutil.copy(img_from, img_to)

       
class utils_file:
    def __init__(self, orig_folder, dest_folder, sec_folder, thr_folder):
        
        self.orig_folder = orig_folder
        self.dest_folder = dest_folder
        self.sec_folder  = sec_folder
        self.thr_folder  = thr_folder

    def counting_class(self, num_of_classes=2):
        '''
        This function counts the number of each class.

        Parameters
        ----------
        counting_folder_path : orig_folder (labels)
            DESCRIPTION.
        num_of_classes : TYPE
            The class that you want to count.

        Returns
        -------
        Number of class object.

        '''
        all_txt_files = glob.glob(self.orig_folder+'/*.txt')
        num_of_files = len(all_txt_files)
        # print(num_of_files)
        count_cls = np.zeros(num_of_classes, dtype=int)
        # print(count_cls)
        number_of_class = 0
        for file in all_txt_files:
            cls_1file = counting_class_in_1file(file, num_of_classes)
            # print(cls_1file)
            count_cls += cls_1file
        print('# of classes in {}: {}'.format(self.orig_folder, count_cls))
        return count_cls

    
    def img_resize(self):
        '''
        Checking images and labels, annotation/main.py would be good.
        But the problem is that it doesn't fit the image if it is larger than 1647 of width pixel.
        This function changes large image sizes to less than 1640.
        Parameters
        ----------
        resize_ratio : 0.85 is the ratio of 19
            DESCRIPTION. The default is 0.8.
            
        self.orig
        self.dest

        Returns
        -------
        None.
        '''
        
        all_img_files = glob.glob(self.orig_folder + '/*.jpg')
        all_files     = len(all_img_files)
        print(all_files)
        for i, img in enumerate(all_img_files):
            print('{}/{}'.format(i, all_files))
            # print(img)
            each_img = cv2.imread(img)
            if each_img.shape[1] >= 1920:
                resize_ratio = 1640 / each_img.shape[1]  # 1647 is max pixel to see full image from "annotation/main.py"
                # print(resize_ratio)
                scale        = (int(each_img.shape[1] * resize_ratio), int(each_img.shape[0] * resize_ratio))
                # print('scale =', scale)
                resized  = cv2.resize(each_img, scale, interpolation = cv2.INTER_AREA)
            else:
                resized = each_img
            cv2.imwrite(os.path.join(self.dest_folder, os.path.basename(img)), resized)
            
            
            
            # print(os.path.basename(img))
            # img_from = img
            # img_to   = os.path.join(out_path, os.path.basename(img))
            # print(img_from)
            # print(img_to)
            # shutil.copy(img_from, img_to)
        
        
    
    def moving_half_of_files(self, file_type_to_move, skip_frame):
        '''
        I guess that images and labels are in the same directory.
        ex) file_type_to_move = 'jpg'
        '''
        for index, file in enumerate(os.listdir(self.orig_folder)):
            print(file)
            name, ext = os.path.splitext(file)
            if ext == file_type_to_move and index % skip_frame == 0:
                print(file)
                
                orig_file = os.path.join(self.orig_folder, file)
                dest_file = os.path.join(self.dest_folder, file)
                
                # orig_label = os.path.join(orig_folder, name+'.txt')
                # dest_label = os.path.join(output_folder, name+'.txt')
                
                # print(orig_label)
                # print(dest_label)
                shutil.move(orig_file,   dest_file)
                # shutil.move(orig_label, dest_label)
        print("Finished!")
        
    def moving_same_name_file(self, file_type='.jpg'):
        '''
        maybe one-time code.
        '''
        print('start!')
        for file in os.listdir(self.orig_folder):
            name, ext = os.path.splitext(file)
            print(file)
            
            orig_img = os.path.join(self.sec_folder, name+'.jpg')
            orig_txt = os.path.join(self.orig_folder, name+'.txt')
            
            dest_img = os.path.join(self.dest_folder, name+'.jpg')
            dest_txt = os.path.join(self.dest_folder, name+'.txt')
            
            shutil.copy(orig_img,   dest_img)
            shutil.copy(orig_txt,   dest_txt)
   
    
   
    # def compareNmove(self):
    #     '''
    #     compare files between orig_folder and sec_folder
    #     and move files in orig_folder to dest_folder.
    #     '''
    #     for 
    

    def moving_non_obj_files(self):
        '''
        This function moves image files that doesn't contain any object based on (or based on) txt files.
        If image doesn't have any object on the txt file, 
        the function moves the image & txt files to the output folder.
        Note that when we have output detected from the yolo model, 
        it contains txt files only for images that has at least one object.
        so, we can then use moving_img_n_lalel function below.
        
        - label_folder : self.orig_folder  (orig_folder)
        - image_folder : containing images (sec_folder)        
        - output_folder: self.dest_folder  (dest_folder)
        '''
        
        # label_folder = self.orig_folder
        # image_folder = self.sec_folder
        # output       = self.dest_folder
        
        # print(image_folder)
        # print(label_folder)
        
        # Reading labels -> v7 detected objects and it saves txt files only when there are objects in the image. 
        # Therefore, we are going to move labels first and then images which has the same name with the labels.
        for label in os.listdir(self.orig_folder):
            
            file_name  = os.path.splitext(label)[0] # current file name. both txt and image.
            image_file = file_name + '.jpg'
            image_full_path = os.path.join(self.sec_folder, image_file)
            # print(image_full_path)
            
            img_dest_full_path = self.dest_folder + '/' + image_file
            
            # print(img_dest_full_path)
            # print(label)
            orig_label = os.path.join(self.orig_folder, label)
            dest_label = os.path.join(self.dest_folder, label)
                       
            print('from :', orig_label)
            # print('to   :', dest_label)
            # print('image_file =', )
            print("size = ", os.stat(orig_label).st_size)
            if os.stat(orig_label).st_size == 0:
                print(orig_label)
                print(dest_label)
                shutil.move(image_full_path, img_dest_full_path)  
                shutil.move(orig_label, dest_label)
                


    def extract_frames_folder(self, frame_interval):
        output_path = os.path.join(self.orig_folder, 'frames')
        
        for file in os.listdir(self.orig_folder):
            file_name, extension = os.path.splitext(file)
            # print(file)
            # print(file_name)
            # print(extension)
            # print('.mp4')
            if extension == '.avi' or '.mp4':
                print('!!!!!!!!!!!')
                each_vid = os.path.join(self.orig_folder, file)
                print('each_vid =', each_vid)
                extract_frames_1video(each_vid, frame_interval, self.dest_folder)
        print('Whole files finished!')
        
    def review_n_move(self):
        '''
        After detecting dataset, selecting falsely detected frames, and moves original frames and labels to a certain folder, so that I can add those dataset to a new dataset.
        self.orig_folder = selected_frames folder.
        self.dest_folder = dest_folder
        self.sec_folder  = orig_img folder
        self.thd_folder  = orig_label folder
        '''
        # reading file name in the selected folder.
        all_img_files = glob.glob(self.orig_folder + '/*.jpg')
        # print(all_img_files)
        for file in os.listdir(self.orig_folder):
            file_name, extension = os.path.splitext(file)
            
            orig_img   = os.path.join(self.sec_folder, file_name+'.jpg')
            orig_label = os.path.join(self.sec_folder, file_name+'.txt')
            # print(orig_img)
            dest_img   = os.path.join(self.dest_folder, file_name+'.jpg')
            dest_label = os.path.join(self.dest_folder, file_name+'.txt')
            
            # shutil.copy(orig_label, dest_label)
            shutil.copy(orig_img, dest_img)
        print('Finished!!!')
        # for img in all_img_files:
        #     print(img)
        
    def moving_not_MF(self):
        '''
        Based on the label files, move non_MF files. i.e. images and labels.
        
        self.orig_folder = labels.
        self.dest_folder = dest
        self.sec_folder  = img folder
        '''
        all_txt_files = glob.glob(self.orig_folder+'/*.txt')
        num_of_files = len(all_txt_files)    
        print(num_of_files)
        for file in os.listdir(self.orig_folder):
            file_name, extension = os.path.splitext(file)
            
            org_label_path  = os.path.join(self.orig_folder, file_name+'.txt')
            org_img_path    = os.path.join(self.sec_folder, file_name+'.jpg')
            dest_label_path = os.path.join(self.dest_folder, file_name+'.txt')
            dest_img_path   = os.path.join(self.dest_folder, file_name+'.jpg')
            
            print(org_label_path)
            # print()
            line_num = 0
            with open(org_label_path, 'r') as file4line:
                for line in file4line:
                    line_num += 1
                    word_num = 0
                    
                    for word in line.split():
                        word_num += 1
                        if word_num == 1:
                            # print(word)
                            if word != '0' and word != '1':#  or '1':
                                # print(line)
                                shutil.copy(org_label_path, dest_label_path)
                                shutil.copy(org_img_path,   dest_img_path)
                                break
                        elif word_num > 1:
                            break

    
        
    def copy_coco_orig_img(self):
        '''
        I have selected only person class in the coco dataset. 
        And I detected male and female classes.
        Now I am going to reannotate them from the original images.
        So I will move original images to the new folder and annotate them again.
        '''
        for file in os.listdir(self.orig_folder):
            file_name, extension = os.path.splitext(file)
            if extension == '.jpg':
                    
                orig_img = self.sec_folder+'/'+file
                out_img  = self.dest_folder+'/'+file
                
                out_label  = self.dest_folder+'/'+file_name+'.txt'
                orig_label = self.sec_folder+'/'+file_name+'.txt'
                
                print('orig_img =', orig_img)
                # print(orig_label)
                
                # print(out_img)
                # print(out_label)
                
                # break
                shutil.copy(orig_img, out_img)
                shutil.copy(orig_label, out_label)
            else:
                continue
            
                
                
    
    def moving_img_n_label(self, based_on_label=False):
        '''
        I want to increase the accuracy with minimum dataset.
        So I am going to add only FNs images and labels.
        It moves all images in the folder and labels with respect to the image names.
        
        When I selected images that the model falsely detected male or female from the detected result images,
        I need to move those images and matching labels from original images and labels.
        Then finally, I correct them to add to a train dataset.
                
        It depends on a situation.
        I changed the code so check the code and folder before using it.        
        
        if not based_on_label:
            self.orig_folder = images and labels are in the folder.
            self.dest_folder = dest
        if based_on_label:
            self.orig_folder = images
            self.dest_folder = output.
            self.thr_folder  = orig images
            self.sec_folder  = orig labels
        
        
        '''
        
        if not based_on_label: # based on images.
            for file in os.listdir(self.orig_folder):
                file_name, extension = os.path.splitext(file)                
                if extension == '.jpg':
                    # print(file_name, extension)
                    # print('self.thr_folder =', self.thr_folder['thr_folder'])
                    # print('file_name       =', file_name)
                    # print(os.path.join(self.thr_folder, file_name))
                    orig_img   = os.path.join(self.thr_folder, file_name) + '.jpg'
                    orig_label = os.path.join(self.sec_folder, file_name) + '.txt'
    
                    dest_img   = os.path.join(self.dest_folder, file_name) + '.jpg'
                    dest_label = os.path.join(self.dest_folder, file_name) + '.txt'
                    
                    print(bool(orig_label))
                    if os.path.exists(orig_label):
                        print('File exists.')
                        shutil.copy(orig_img, dest_img)
                        shutil.copy(orig_label, dest_label)
                    else:
                        print("Label doesn't exists.")
                        shutil.copy(orig_img, dest_img)
        else:
            for file in os.listdir(self.orig_folder):
                file_name, extension = os.path.splitext(file)
                if extension == '.txt':
                    print(file)
                    orig_img   = os.path.join(self.orig_folder, file_name) + '.jpg'
                    orig_label = os.path.join(self.orig_folder, file_name) + '.txt'
    
                    dest_img   = os.path.join(self.dest_folder, file_name) + '.jpg'
                    dest_label = os.path.join(self.dest_folder, file_name) + '.txt'
                    
                    print(bool(orig_label))
                    if os.path.exists(orig_label):
                        print('File exists.')
                        shutil.copy(orig_img, dest_img)
                        shutil.copy(orig_label, dest_label)
                    else:
                        print("Label doesn't exists.")
                        shutil.copy(orig_img, dest_img)


    def move_from_file_name(self, extension):
        """
        It moves images and labels by reading any file name with specific extension in the folder.
        It also can select folders of files we want to move from, and label folder.
        
        file_orig = file names from

        orig_img   = os.path.join(self.sec_folder, file_name) + '.jpg'
        orig_label = os.path.join(self.thr_folder, file_name) + '.txt'
        
        dest_img   = os.path.join(self.dest_folder, file_name) + '.jpg'
        dest_label = os.path.join(self.dest_folder, file_name) + '.txt'
        """
        # len_all_files = len(glob.glob(self.orig_folder+'/*.*'))

        for file in os.listdir(self.orig_folder):
            file_name, ext = os.path.splitext(file)
            # print(file_name)
            # print(ext)
            if ext == extension:
                
                orig_img   = os.path.join(self.sec_folder, file_name) + '.jpg'
                orig_label = os.path.join(self.thr_folder, file_name) + '.txt'
                
                # print('orig_label =', orig_label)
                dest_img   = os.path.join(self.dest_folder, file_name) + '.jpg'
                dest_label = os.path.join(self.dest_folder, file_name) + '.txt'
                
                if os.path.exists(orig_img):
                    # print('Image exists.')
                    shutil.copy(orig_img, dest_img)
                    # shutil.move(orig_img, dest_img)
                    if os.path.exists(orig_label):
                        # print('Label exists.')                
                        shutil.copy(orig_label, dest_label)
                        # shutil.move(orig_label, dest_label)
                    # else:
                        # print('Label does not exit. Pass.')
                else:
                    print("file_name =", file_name)
                    print("Both Image and Label don't exists. Next.")
        print("Finished.")
                    
    def png_to_jpg(self):
        for file in os.listdir(self.orig_folder):
            file_name, ext = os.path.splitext(file)
            if ext == ".png":
                file_read  = os.path.join(self.orig_folder,file)
                file_write = os.path.join(self.dest_folder,file_name+ ".jpg")
                # print(file_write)
                img = cv2.imread(file_read)
                cv2.imwrite(file_write, img)
                
    def change_class(self):
        '''
        I used this function to change all classes to one class.
        '''
        all_txt_files = glob.glob(self.orig_folder+'/*.txt')
        num_of_files  = len(all_txt_files)    
        print(num_of_files)
        for file in os.listdir(self.orig_folder):
            file_name, extension = os.path.splitext(file)
            # print(file_name)
            org_label_path  = os.path.join(self.orig_folder, file_name+'.txt')
            # org_img_path    = os.path.join(self.sec_folder, file_name+'.jpg')
            dest_label_path = os.path.join(self.dest_folder, file_name+'.txt')
            # dest_img_path   = os.path.join(self.dest_folder, file_name+'.jpg')
            
            print(org_label_path)
            print(dest_label_path)
            # print()
            
            with open(org_label_path, 'r') as file4line:
                lines     = file4line.readlines()
            
            with open(dest_label_path, 'w') as new_file:
            # lines.split()[1]
                for line in lines:                   # Reading each line.
                    print(line)
                    word_num = 1
                    # for word in line.split():        # Reading each word.
                    line_write = line.split()
                    
                    # print(line_write1)
                    if line_write[0] == str(0):   # if class == 0
                        line_write[0] = str(1)    # change class to "1".
                    line_write1 = str(line_write[0]) + " " + str(line_write[1]) + " " + str(line_write[2]) + " " + str(line_write[3]) + " " + str(line_write[4] + "\n")
                        
                    new_file.write(line_write1)
                    # break
        
                
    def csv_creator_from_yolo(self):
        '''
        pytorch dataframe requires csv format labels.
        First, I cropped each class bjects from images.
        And put them into each folder that is named as their class.
        Now I am going to make csv file which contains file name and class.
        In 'orig' folder, there are 'M' and 'F' folders.
        '''
        male_folder   = os.path.join(self.orig_folder, 'M') 
        female_folder = os.path.join(self.orig_folder, 'F')
        
        with open(r'D:\safety_2022\Gender_Classification\2023_spring_conference_weights\cropped_train\train.csv', mode='w', newline='') as csv_file:
            writerObj = csv.writer(csv_file)
            writerObj.writerow(['image_id', 'label'])
            for file in os.listdir(male_folder):
                print(os.path.abspath(file))
                # content = [os.path.abspath(file), 'Male']
                
                # 0 for F
                # 1 for M
                content = [file, '1']
                writerObj.writerow(content)
            for file in os.listdir(female_folder):
                print(os.path.abspath(file))
                content = [file, '0']
                writerObj.writerow(content)
    
    def file_rename(self, prefix = 'SSD_'):
        # all_files = glob.glob(self.orig_folder+'/*.txt')
        # # num_of_files  = len(all_txt_files)    
        # # print(num_of_files)
        for file in os.listdir(self.orig_folder):
            # print(file)
            orig_name = os.path.join(self.orig_folder, file)
            # print(full_file)
            new_name = os.path.join(self.orig_folder, prefix + file)
            # print(new_name)
            os.rename(orig_name, new_name)
            # file_name, ext = os.path.splitext(file)
            # print(file_name)
        
    def frame_extractor_btwn_time(self, before=14, duration=10, frame_interval=8):
        '''
        This extracts video frames from start time.
        We can skip to 20 [sec] using cap.set(cv2.CAP_PROP_POS_MSEC, 20000).
                
        before   in sec. -> start time is when a person falls down. So we need frames before a person falldown. 3 [sec].
                            until 9 [sec].
        start_time in sec.
        orig: includes xml files
        dest: outputs frames.
        '''
        all_xml_files = glob.glob(self.orig_folder + '/*.xml')
        # print(all_xml_files)
        for xml_file in all_xml_files:
            print(xml_file)
            start_time = start_time_from_xml(xml_file)
            # print(os.path.basename(xml_file), start_time)
            start_time = start_time - before
            # print(os.path.splitext(os.path.basename(xml_file))[0])
            end_time = start_time + duration
           
            
            video_path     = os.path.join(self.orig_folder, os.path.splitext(os.path.basename(xml_file))[0] +  '.mp4')
            
            # print(video_path)v
            # break
            extract_frames_1video_period(video_path, self.dest_folder, start_time, end_time, frame_interval)
            # break

    def renamer(self, remove_letter):
        all_img_files = glob.glob(self.orig_folder + '/*.jpg')
        len_all_files = len(all_img_files)
        for i, file in enumerate(all_img_files):
            print('{}/{}'.format(i, len_all_files), end='\r')
            orig_file = os.path.join(self.orig_folder, file)

            only_file_name = os.path.splitext(file)[0]
            only_file_name = only_file_name.replace(remove_letter, '')
            # print(only_file_name)
            new_file  = os.path.join(self.orig_folder, only_file_name+os.path.splitext(file)[1])
            # print('orig =', orig_file)
            # print('new  =', new_file)
            os.rename(orig_file, new_file)



    def moving_files(self, skip_files=10):
        all_img_files = glob.glob(self.orig_folder + '/*.jpg')
        print(all_img_files)
        for i, file in enumerate(all_img_files):
            # print(file)
            base_name = os.path.basename(file)
            # print(base_name)
            # only_file_name = os.path.splitext(base_name)
            # print(only_file_name)
            new_file  = os.path.join(self.dest_folder, base_name)
            # print(new_file)
            if (i%skip_files) == 0:
                shutil.move(file, new_file)
            i += 1
            
    def crop_object(self):
        '''
        It crops object from orignal images by reading yolo output coordinates.
        
        orig_img: orig_folder
        label   : thr_folder        

        Returns
        -------
        None.

        '''
        all_images = glob.glob(self.orig_folder + '/*.jpg')
        all_labels = glob.glob(self.thr_folder + '/*.txt')
        
        for i, label in enumerate(all_labels):
            # print(i, label)
            base_name      = os.path.basename(label)
            only_file_name = os.path.splitext(base_name)[0]
            # print(only_file_name)
            image_path = os.path.join(self.orig_folder, only_file_name+'.jpg')
            image      = cv2.imread(image_path)
            
            # print(image)
            
            line_num = 0
            # opening file.
            with open(label, 'r') as file4line:
                
                # reding each line    
                for line in file4line:
                    print('line =', line)
                    line_num += 1
                    
                    this_line  = line.split()
                    cls_label  = int(this_line[0])
                    x_cen, y_cen, width, height = map(float, this_line[1:])
                    # print(cls_label,x,y,w,h)
                    
                    if cls_label == 0: # falling.
                        x, y, w, h = int((x_cen - width / 2) * image.shape[1]), int((y_cen - height / 2) * image.shape[0]), int(width * image.shape[1]), int(height * image.shape[0])
                        print(x,y,w,h)
                        cropped_obj = image[y:y+h, x:x+w]
                        out_file_name = os.path.join(self.dest_folder, only_file_name + '_' + str(line_num) + '.jpg')
                        print(out_file_name)
                        cv2.imwrite(out_file_name, cropped_obj)
        
    
    def mix_img(self, where2put='bottom'):
        """
        When we have FP images, we need to add some object to those FP-images to re-train the model.
        So that the model does not ignore these FP-images when training.
        This function adds a small object image to the orig FP-images.
        
        Parameters
        ----------
        where2put : str
            DESCRIPTION. The default is 'bottom'.
            This is where we paste the small image to the FP-images.
            
            orig        = orig: large images.
            dest_folder = dest
            sec_folder  = obj: small images

        Returns
        -------
        None.
        """
        
        # reading all images in the FPs.
        all_FP_img_files = glob.glob(self.orig_folder + '/*.jpg')
        all_obj_files    = glob.glob(self.sec_folder + '/*.jpg')
        # print(len(all_FP_img_files))
        # print(len(all_obj_files))
        # print(all_obj_files[1])
        if len(all_FP_img_files) > len(all_obj_files):
            print("Not enough object images. Add more object images. Break.")
            
        for i, _ in enumerate(all_FP_img_files):
            # print('i =', i)
            FN_img  = cv2.imread(all_FP_img_files[i])
            obj_img = cv2.imread(all_obj_files[i])
            
            out_path = os.path.join(self.dest_folder, os.path.split(all_FP_img_files[i])[1])
            
            FN_img_w  = FN_img.shape[1]
            FN_img_h  = FN_img.shape[0]
            obj_img_w = obj_img.shape[1]
            obj_img_h = obj_img.shape[0]
            # print(FN_img.shape)
            # print(obj_img.shape)
            
            
            if where2put == 'bottom':
                print("* Adding to the bottom *")
                
                # checking if obj img is larger than height.
                # if it is, then reduce to half of the size.
                if obj_img_h >= FN_img_h/2:
                    print(all_obj_files[i], "This object image is too big. Reducing to half.")
                    # obj_img = cv2.resize(obj_img, (int(obj_img_w/2), int(obj_img_h/2)))
                    obj_img = cv2.resize(obj_img, (int(200*obj_img_w/obj_img_h), 200))
                    
                    obj_img_w = obj_img.shape[1]
                    obj_img_h = obj_img.shape[0]

                start_p_w = np.random.randint(0, (FN_img_w  - obj_img_w))
                start_p_h = np.random.randint((FN_img_h / 2), FN_img_h - obj_img_h)
                end_p_w = start_p_w + obj_img_w
                end_p_h = start_p_h + obj_img_h
                
                FN_img[start_p_h:end_p_h, start_p_w:end_p_w, :] = obj_img
                
                cv2.imwrite(out_path, FN_img)
            
            if where2put == 'top':
                print("* Adding to the top *")
                
                # checking if obj img is larger than height.
                # if it is, then reduce to half of the size.
                if obj_img_h >= FN_img_h/2:
                    print(all_obj_files[i], "This object image is too big. Reducing to half.")
                    # obj_img = cv2.resize(obj_img, (int(obj_img_w/2), int(obj_img_h/2)))
                    obj_img = cv2.resize(obj_img, (int(200*obj_img_w/obj_img_h), 200))
                    
                    obj_img_w = obj_img.shape[1]
                    obj_img_h = obj_img.shape[0]

                start_p_w = np.random.randint(0, (FN_img_w  - obj_img_w))
                start_p_h = np.random.randint(0, FN_img_h/2 - obj_img_h)
                end_p_w = start_p_w + obj_img_w
                end_p_h = start_p_h + obj_img_h
                
                FN_img[start_p_h:end_p_h, start_p_w:end_p_w, :] = obj_img
                
                cv2.imwrite(out_path, FN_img)
                
            if where2put == 'right':
                print("* Adding to the right *")
                
                # checking if obj img is larger than height.
                # if it is, then reduce to half of the size.
                if obj_img_h >= FN_img_h/2:
                    print(all_obj_files[i], "This object image is too big. Reducing to half.")
                    # obj_img = cv2.resize(obj_img, (int(obj_img_w/2), int(obj_img_h/2)))
                    obj_img = cv2.resize(obj_img, (int(200*obj_img_w/obj_img_h), 200))
                    
                    obj_img_w = obj_img.shape[1]
                    obj_img_h = obj_img.shape[0]

                start_p_w = np.random.randint(FN_img_w/2, (FN_img_w  - obj_img_w))
                start_p_h = np.random.randint(0, obj_img_h)
                end_p_w = start_p_w + obj_img_w
                end_p_h = start_p_h + obj_img_h
                
                FN_img[start_p_h:end_p_h, start_p_w:end_p_w, :] = obj_img
                
                cv2.imwrite(out_path, FN_img)
            
            if where2put == 'left':
                print("* Adding to the right *")
                
                # checking if obj img is larger than height.
                # if it is, then reduce to half of the size.
                if obj_img_h >= FN_img_h/2:
                    print(all_obj_files[i], "This object image is too big. Reducing to half.")
                    # obj_img = cv2.resize(obj_img, (int(obj_img_w/2), int(obj_img_h/2)))
                    obj_img = cv2.resize(obj_img, (int(200*obj_img_w/obj_img_h), 200))
                    
                    obj_img_w = obj_img.shape[1]
                    obj_img_h = obj_img.shape[0]

                start_p_w = np.random.randint(0, (FN_img_w/2  - obj_img_w))
                start_p_h = np.random.randint(0, obj_img_h)
                end_p_w = start_p_w + obj_img_w
                end_p_h = start_p_h + obj_img_h
                
                FN_img[start_p_h:end_p_h, start_p_w:end_p_w, :] = obj_img
                
                cv2.imwrite(out_path, FN_img)    
            
            i += 1
        
        # cv2.imshow('ddd', FN_img)
        # # cv2.imshow('ddd', obj_img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
            
        
        
    # def img_resize(self):
    #     all_img_files = glob.glob(self.orig_folder + '/*.jpg')
    #     all_files = len(all_img_files)
    #     print(all_files)
    #     for i, img in enumerate(all_img_files):
    #         print('{}/{}'.format(i, all_files))
            
    #         each_img = cv2.imread(img)
    #         scale = (200, 200)
    #         resized  = cv2.resize(each_img, scale, interpolation = cv2.INTER_AREA)
    #         cv2.imwrite(os.path.join(self.dest_folder, os.path.basename(img)), resized)



if __name__ == '__main__':
    
    orig        = r'D:\Seoulmetro\safety_2022\KISA_certification\dataset\training_dataset\dataset_231128\train\images'
    dest_folder = r'C:\Users\ossam\Desktop\annotating\final_images\right'
    sec_folder  = r'C:\Users\ossam\Desktop\annotating\cropped_obj_selected\right'
    thr_folder  = r''
    
    # orig        = r'C:\Users\ossam\OneDrive\Desktop\willbeadded\FPs_empty\both'
    # dest_folder = r'C:\Users\ossam\OneDrive\Desktop\willbeadded\FPs_empty\orig_both'
    # sec_folder  = r'D:\safety_2022\KISA_certification\dataset\kisa_DB\whole_frames'
    # thr_folder  = r''
    
    uf = utils_file(orig_folder=orig, dest_folder=dest_folder, sec_folder=sec_folder, thr_folder=thr_folder)

    # uf.moving_files(skip_files=11)

    # uf.frame_extractor_btwn_time()
    
    # uf.png_to_jpg()
    
    # uf.crop_images(180, 1080, 0, 1920)
    
    # uf.moving_half_of_files(file_type_to_move='.jpg', skip_frame=3)
    
    # uf.file_rename(prefix='SSD_')
    
    # uf.moving_same_name_file()
    
    # moving_half_of_files(orig, out)
    
    uf.counting_class()
        
    # fps_check(r'D:\Github\DUP\dataset\cropped.mp4')
    
    # uf.review_n_move()

    # uf.moving_not_MF()

    # uf.copy_coco_orig_img()
    
    # crop_1image(r'D:\my_doc\safety_2022\videos\jegidong\jegidong_shutter_20221205_13_16\splitted\jegidong_shutter_20221205_13_16_000_000060.jpg', r'D:\my_doc\safety_2022\videos\jegidong\jegidong_shutter_20221205_13_16\splitted\frames_crop')
    
    # uf.moving_img_n_label(based_on_label=True)
    
    # uf.change_class()
    
    # uf.csv_creator_from_yolo()
    
    # uf.img_resize()


    #%% Making dataset.
    # uf.extract_frames_folder(10)
    # uf.moving_non_obj_files()
    # uf.renamer(remove_letter='_orig')
    # uf.move_from_file_name(extension='.jpg') # This would be frequently used.
    # uf.crop_object()
    # uf.mix_img(where2put='right')     # patching obj image to the orig img.
    
    
    # uf.img_resize()
    


# video_path = r'D:\safety_2022\KISA_certification\dataset\kisa_dataset\research_data\abroad_environment_1500\falldown_330\C058303_001.mp4'
# output_dir = r'D:\safety_2022\KISA_certification\dataset\falldown_data\temp'
# xml_folder = r'D:\safety_2022\KISA_certification\dataset\kisa_dataset\research_data\abroad_environment_1500\falldown_330'

# start_time = 30 
# end_time   = 40 
# # extract_frames_1video_period(video_path, output_dir, start_time, end_time)

# loading_xml_time(xml_folder)
