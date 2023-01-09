import os
import glob
import shutil
import numpy as np
import cv2
from multiprocessing import Pool


def test_again():
    print('Will this be synched???')
    print('Will this be synched???')
    pass
    

def crop_images(img_path, output_path, h1,h2,w1,w1): # ratio = h1:h2, w1:w2   , add "start_point, end_point," next time 
    
    all_img_files = glob.glob(img_path + '/*.jpg')
    # print(all_img_files)
    for img in all_img_files:
        print(img)
        crop_1image(img, output_path, h1,h2,w1,w1)
    print('Finished!!!')
    

def crop_1image(image_path, out_path, ratio):
    '''
    crop ratio is the ratio from the top and left of the image to be removed from the orig img.
    '''
    
    img      = cv2.imread(image_path)
    # i_w = img.shape[1]
    # i_h = img.shape[0]
    
    base_name = os.path.basename(image_path)
    # print(crop_ratio)
    crop_img = img[h1:h2,w1:w1]
    
    

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
    
    frame_width  = int(cap.get(3))
    frame_height = int(cap.get(4))
    fps          = round(cap.get(cv2.CAP_PROP_FPS)) # current frame = cap.get(1)
    
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
    def __init__(self, orig_folder, dest_folder, sec_folder, **thr_folder):
        
        self.orig_folder = orig_folder
        self.dest_folder = dest_folder
        self.sec_folder  = sec_folder
        self.thr_folder  = thr_folder

    def counting_class(self, num_of_classes=3):
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

    
    def img_resize(self, resize_ratio=0.7):
        all_img_files = glob.glob(self.orig_folder + '/*.jpg')
        all_files     = len(all_img_files)
        print(all_files)
        for i, img in enumerate(all_img_files):
            print('{}/{}'.format(i, all_files))
            each_img = cv2.imread(img)
            scale    = (int(each_img.shape[1] * resize_ratio), int(each_img.shape[0] * resize_ratio))
            resized  = cv2.resize(each_img, scale, interpolation = cv2.INTER_AREA)
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
        This function moves image files that doesn't contain any object from (or based on) txt files.
        If image doesn't have any object on the txt file, 
        the function moves the image & txt files to the output folder.
        
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
                       
            # print('from :', orig_label)
            # print('to   :', dest_label)
            # print('image_file =', )
            if os.stat(orig_label).st_size != 0:
                print(orig_label)
                  
                shutil.move(orig_label, dest_label)
                shutil.move(image_full_path, img_dest_full_path)


    def extract_frames_folder(self, frame_interval):
        output_path = os.path.join(self.orig_folder, 'frames')
        # print(path)
        for file in os.listdir(self.orig_folder):
            file_name, extension = os.path.splitext(file)
            # print(file)
            # print(file_name)
            # print(extension)
            # print('.mp4')
            if extension == '.avi' or '.mp4':
                print('!!!!!!!!!!!')
                each_vid = os.path.join(self.orig_folder, file)
                print(each_vid)
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
            
            shutil.copy(orig_label, dest_label)
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
                                shutil.copy(org_img_path, dest_img_path)
                                break
                        elif word_num > 1:
                            break

    def moving_img_n_label(self):
        '''
        I want to increase the accuracy with minimum dataset. So I am going to add add only FNs images and labels.
        It moves all images in the folder and labels with respect to the image names.
        
        It depends on the situation.
        I changed the code so check the code and folder before using it.        

        self.orig_folder = images and labels are in the folder.
        self.dest_folder = dest
        '''
        for file in os.listdir(self.orig_folder):
            file_name, extension = os.path.splitext(file)                
            if extension == '.jpg':
                # print(file_name, extension)
                orig_img   = os.path.join(self.sec_folder, file_name) + '.jpg'
                orig_label = os.path.join(self.sec_folder, file_name) + '.txt'

                dest_img   = os.path.join(self.dest_folder, file_name) + '.jpg'
                dest_label = os.path.join(self.dest_folder, file_name) + '.txt'
                
                # print(bool(orig_label))
                if os.path.exists(orig_label):
                    print('File exists.')
                    shutil.move(orig_img, dest_img)
                    shutil.move(orig_label, dest_label)
                else:
                    print("Label doesn't exists.")
                    shutil.move(orig_img, dest_img)



                            
                        
                    
    
if __name__ == '__main__':
    
    orig       = r'D:\my_doc\safety_2022\videos\paths\changdong\changdong4_path_south_221110_17_20\splitted\frames'
    out        = r'D:\my_doc\safety_2022\videos\paths\changdong\changdong4_path_south_221110_17_20\splitted\frames\cropped'
    sec_folder = r''
    thr_folder = r''
    
    uf = utils_file(orig_folder=orig, dest_folder=out, sec_folder=sec_folder, thr_folder=thr_folder)
    uf.crop_images(180, 1080, 0, 1920)
    # uf.moving_half_of_files(file_type_to_move='.jpg', skip_frame=3)
    # uf.img_resize()
    # uf.moving_same_name_file()
    # uf.moving_non_obj_files()
    
    # moving_half_of_files(orig, out)
    # uf.counting_class()
    
    # fps_check(r'D:\my_doc\safety_2022\videos\platform\euljiro\splitted\euljiro_20221101_17_20_000.mp4')
    # uf.extract_frames_folder(120)
    
    # uf.review_n_move()
    # uf.moving_not_MF()

    # uf.moving_img_n_label()
    # crop_1image(r'D:\my_doc\safety_2022\videos\jegidong\jegidong_shutter_20221205_13_16\splitted\jegidong_shutter_20221205_13_16_000_000060.jpg', r'D:\my_doc\safety_2022\videos\jegidong\jegidong_shutter_20221205_13_16\splitted\frames_crop')
    
