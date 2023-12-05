# -*- coding: utf-8 -*-

import os
# os.add_dll_directory(os.getcwd())
import vlc
import time
# import bs4
import cv2
import datetime
import numpy as np
from shapely.geometry import Point, Polygon
# from collections import deque
from bs4 import BeautifulSoup

def curr_time(curr_time_sec):
    # input: curr_time_sec = player.get_time()/1000
    # return: min and sec
    curr_time     = divmod(curr_time_sec, 60)
    curr_min      = int(curr_time[0])
    curr_sec      = int(curr_time[1])
    return curr_min, curr_sec

def curr_vid_time(url="rtsp://192.168.0.2:8554/"):
    vlc_instance = vlc.Instance()
    player       = vlc_instance.media_player_new()
    media        = vlc_instance.media_new(url) # "rtsp://192.168.0.2:8554/"
    player.set_media(media)

    # Start playing the video
    player.play()
    
    # Wait for the video to start playing
    time.sleep(3)  # Adjust the sleep time as needed
    curr_time_sec = player.get_time()/1000
    curr_time     = divmod(curr_time_sec, 60)
    curr_min      = int(curr_time[0])
    curr_sec      = int(curr_time[1])
    # print("Current playback time :", curr_min, '[min]', curr_sec, '[sec]')
    player.stop()
    return curr_min, curr_sec, curr_time_sec


def load_mp4_file(xml_file):
    with open(xml_file, 'r') as f:
        data = f.read()
    bs_data = BeautifulSoup(data, 'xml')
    print(bs_data)


def xml_checker_n_fixer(map_file):
    with open(map_file, 'r') as f:
        first_line = next(f)
        # print('first line =', first_line)
        if 'ï»¿' in first_line:
            # print("The first line includes ï»¿. --> encoding='utf-8-sig'")
            with open(map_file, 'r', encoding="utf-8-sig") as f:
                data = f.read()
            # print('data =', data)
            return data        
        else:
            # print("The first line does not include ï»¿ --> normal encoding.")
            with open(map_file, 'r') as f:
                data = f.read()
            return data
    
    

    
    
def load_map(video_name='', map_folder=r'C:\Users\ossam\OneDrive\Desktop\KISA\MAP'):
    map_folder = r'C:\Users\ossam\OneDrive\Desktop\KISA\MAP_abroad_added'
    '''
    From video file name, it recall ROI area from map file.
    '''
    map_file = video_name[0:7] + '.map'
    map_file = os.path.join(map_folder, map_file)
    
    data = xml_checker_n_fixer(map_file)
    
    # reading all xml data.
    bs_data           = BeautifulSoup(data, "xml") 
    
    # print('bs_data =', bs_data)
    # How many DetectionAreas. int.
    DetectionAreas    = int(bs_data.find_all('DetectionAreas')[0].text)
    # print(DetectionAreas)
    # DetectArea points.
    DetectArea_points = bs_data.find('DetectArea')
    if DetectArea_points:
        DetectArea_points = DetectArea_points.find_all('Point') 
        DetectArea        = [list(map(int, point.text.split(','))) for point in DetectArea_points]
        DetectArea_tup    = [(x, y) for x, y in DetectArea]
    else:
        DetectArea = None
    
    # Loitering points.
    Loitering_points = bs_data.find('Loitering')
    if Loitering_points:
        Loitering_points = Loitering_points.find_all('Point')
        Loitering        = [list(map(int, point.text.split(','))) for point in Loitering_points]
        Loitering_tup    = [(x, y) for x, y in Loitering]
        DetectArea = None
    else:
        Loitering     = None
        Loitering_tup = None
        
    # Intrusion points.
    Intrusion_points = bs_data.find('Intrusion')
    if Intrusion_points:
        Intrusion_points = Intrusion_points.find_all('Point')
        Intrusion        = [list(map(int, point.text.split(','))) for point in Intrusion_points]
        Intrusion_tup    = [(x, y) for x, y in Intrusion]
        DetectArea = None
    else:
        Intrusion     = None
        Intrusion_tup = None
        
    return DetectionAreas, DetectArea, Intrusion, Loitering, DetectArea_tup, Intrusion_tup, Loitering_tup #, map_file


def Drawing_boundaries(img, DetectionAreas, DetectArea, Loitering, Intrusion):
    # img = cv2.imread(img)
    if DetectArea:
        DetectArea_pts = np.array(DetectArea, np.int32)
        # print(DetectArea_pts)
        DetectArea_pts = DetectArea_pts.reshape((-1, 1, 2))
        # print(DetectArea_pts)
        image = cv2.polylines(img, 
                              [DetectArea_pts], 
                              isClosed=True, 
                              color=(0,255,0), 
                              thickness=8)
    if Loitering:
        Loitering_pts = np.array(Loitering, np.int32)
        # print(Loitering_pts)
        Loitering_pts = Loitering_pts.reshape((-1, 1, 2))
        # print(Loitering_pts)
        image = cv2.polylines(image, 
                              [Loitering_pts], 
                              isClosed=True, 
                              color=(255,0,0), 
                              thickness=8)
    if Intrusion:
        Intrusion_pts = np.array(Intrusion, np.int32)
        # print(DetectArea_pts)
        Intrusion_pts = Intrusion_pts.reshape((-1, 1, 2))
        # print(DetectArea_pts)
        image = cv2.polylines(img, 
                              [Intrusion_pts], 
                              isClosed=True, 
                              color=(0,0,255), 
                              thickness=8)
    
    # cv2.imshow('iii', image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()


def is_point_inside_polygon(x, y, polygon):
    n        = len(polygon) # number of vertices
    inside   = False
    p1x, p1y = polygon[0]   # setting the first vertice of polygon.
    # print(polygon[0])
    for i in range(n+1):          # loop for each vertices of polygon. from start to start.
        # print(i)
        p2x, p2y = polygon[i % n] # loop for each vertices. from start to start.
        if y     >  min(p1y, p2y):  # if y is between p2y and p1y,
            if y <= max(p1y, p2y): 
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
        p1x, p1y = p2x, p2y
    return inside


def rect_in_poly_3points(rectangle, polygon):
    rectangle_np = [(rectangle[0].item(), rectangle[1].item()),
                    (rectangle[2].item(), rectangle[1].item()),
                    (rectangle[0].item(), rectangle[3].item()),
                    (rectangle[2].item(), rectangle[3].item())]
    # print('rectangle =', rectangle_np)
    # print('polygon   =', polygon)
    poly = Polygon(polygon)
    for point in rectangle_np: # loop for each vertices of rectangle.
        if not Point(point).within(poly):
            return False
    return True


def cen_3_4_in_poly(rectangle, polygon):
    cen_3_4 = (  (rectangle[0].item() + rectangle[2].item())/2,  rectangle[1].item() + ((rectangle[3].item() - rectangle[1].item())*4/5)   )
    # print('cen_2_3 =', cen_2_3)
    poly    = Polygon(polygon)
    if Point(cen_3_4).within(poly):
        return True
    return False

def center_in_poly(rectangle, polygon):
    center  = (  (rectangle[0].item() + rectangle[2].item())/2,  (rectangle[1].item() + rectangle[3].item())/2   )
    
    # print('center =', center)
    # rectangle_np = [(rectangle[0].item(), rectangle[1].item()),
    #                 (rectangle[2].item(), rectangle[1].item()),
    #                 (rectangle[0].item(), rectangle[3].item()),
    #                 (rectangle[2].item(), rectangle[3].item())]
    
    # center = (  (xyxy[0]+xyxy[2])/2, (xyxy[1]+xyxy[3])/2  )
    poly = Polygon(polygon)
    if Point(center).within(poly):
        # print('point is in the poly!!!')
        return True
    # print('point is NOT in the poly!!!')
    return False

def rect_in_poly(rectangle, polygon):
    rectangle_np = [(rectangle[0].item(), rectangle[1].item()),
                    (rectangle[2].item(), rectangle[1].item()),
                    (rectangle[0].item(), rectangle[3].item()),
                    (rectangle[2].item(), rectangle[3].item())]
    # print('rectangle =', rectangle_np)
    # print('polygon   =', polygon)
    poly = Polygon(polygon)
    for point in rectangle_np: # loop for each vertices of rectangle.
        if not Point(point).within(poly):
            return False
    return True

def in_poly(rectangle, polygon, file_name):
    file            = file_name[0:7]
    ex_center_list  = ['C001102', 'C110101', 'C110201', 'C110301']
    ex_2_3_list     = ['C107101', 'C107201', 'C107301']
    if file in ex_center_list:
        # print(file, ' is in ex_center_list.')
        return center_in_poly(rectangle, polygon)
    elif file in ex_2_3_list:
        # print(file, ' is in ex_2_3_list.')
        return cen_3_4_in_poly(rectangle, polygon)
    else:
        # print(file, ' is not in the list. Rect in poly.')
        return rect_in_poly(rectangle, polygon)


def xml_writer(input_file  = "C002200_002.mp4", 
               out_folder  = r"C:\Users\ossam\OneDrive\Desktop\SA",
               intrusion_t = None,     loitering_t=None,     falldown_t=None,
               intrusion_d = "0:1:00", loitering_d="0:1:00", falldown_d="0:1:00",
               all_events  = ['Intrusion', 'Loitering', 'Falldown']):
    
    '''
    2 inputs: input_file and _t
    '''
    out_file = os.path.splitext(input_file)[0] + '.xml'
    out_file = os.path.join(out_folder, out_file)
    # print(out_file)

    # counting number of events happened in the video.
    
    all_events_time     = [intrusion_t, loitering_t, falldown_t]
    all_events_duration = [intrusion_d, loitering_d, falldown_d]
    
    # creating pair of event happened and its time. [[time, event]]
    events_pair     = [[time, event, duration] for time, event, duration in zip(all_events_time, all_events, all_events_duration) if time is not None]
    # print('events_pair =', events_pair)
    # events_pair = [['0:2:63', 'Loitering', '0:1:00']]

    # sorting events in time.
    sorted_pair = sorted(events_pair, key = lambda x : datetime.datetime.strptime(x[0], "%H:%M:%S"), reverse=False)
    # print('sorted_pair =', sorted_pair)
    num_events = len(sorted_pair)

    # Create a new XML structure using BeautifulSoup
    soup = BeautifulSoup(features="xml")
    
    kisa_library_index = soup.new_tag("KisaLibraryIndex")
    soup.append(kisa_library_index)
    
    library = soup.new_tag("Library")
    kisa_library_index.append(library)
    
    clip = soup.new_tag("Clip")
    library.append(clip)
    
    header = soup.new_tag("Header")
    clip.append(header)
    
    alarm_events = soup.new_tag("AlarmEvents")
    alarm_events.string = str(num_events)
    header.append(alarm_events)
    
    filename = soup.new_tag("Filename")
    filename.string = input_file
    header.append(filename)
    
    alarms = soup.new_tag("Alarms")
    clip.append(alarms)
    
    for index in range(num_events):
        # print(index)
        # print(sorted_pair[index])
        
        alarm = soup.new_tag("Alarm")
        alarms.append(alarm)
        
        start_time = soup.new_tag("StartTime")
        start_time.string = sorted_pair[index][0]
        alarm.append(start_time)
        
        alarm_description = soup.new_tag("AlarmDescription")
        alarm_description.string = sorted_pair[index][1]
        alarm.append(alarm_description)
        
        alarm_duration = soup.new_tag("AlarmDuration")
        alarm_duration.string = sorted_pair[index][2]
        alarm.append(alarm_duration)
    
    # Write the XML structure to a file with line breaks
    with open(out_file, "w") as file:
        file.write(str(soup))


def are_all_same(numbers):
    if len(numbers) == 0:
        return True
    else:
        first_number = numbers[0]
        return all(number == first_number for number in numbers)


def file_list(raw_txt_file, output_file_name):
    """
    Creating KISA file list in xml form from txt file.
    """
    
    # reading each line
    with open(raw_txt_file, 'r') as file4line:
        file_names = file4line.read().splitlines()
        
    # Create the XML structure
    soup                = BeautifulSoup(features="xml")
    streaming_file_list = soup.new_tag("StreamingFileList")
    soup.append(streaming_file_list)
    
    list_format_version        = soup.new_tag("ListFormatVersion")
    list_format_version.string = "1.0"
    streaming_file_list.append(list_format_version)
    
    num_of_files        = soup.new_tag("NumOfFiles")
    num_of_files.string = str(len(file_names))
    streaming_file_list.append(num_of_files)
    
    files = soup.new_tag("Files")
    streaming_file_list.append(files)
    
    for file_name in file_names:
        file_tag = soup.new_tag("File")
        files.append(file_tag)
        
        name_tag = soup.new_tag("Name")
        name_tag.string = file_name
        file_tag.append(name_tag)

    with open(output_file_name, 'w') as file_list:
        file_list.write(str(soup.prettify()))
        
            
def reading_file_list(xml_file):
    '''
    # It returns file_names in the list from the xml file.    
    '''
    # Read the XML file
    with open(xml_file, 'r') as file:
        xml_data = file.read()
    
    # Parse the XML data with BeautifulSoup
    soup = BeautifulSoup(xml_data, 'xml')
    
    # Extract file names
    file_names = []
    file_tags = soup.find_all('File')
    for file_tag in file_tags:
        name_tag = file_tag.find('Name')
        if name_tag:
            file_name = name_tag.text.strip()
            file_names.append(file_name)
    
    # print(file_names)
    # # Print the file names
    # i=1
    # for file_name in file_names:
    #     print('file name =', file_name)
    #     i +=1
    return file_names

def removing_file_n_save_again(xml_data, xml_file):
    # print(xml_data)
    soup = BeautifulSoup(xml_data, 'xml')
    # print('soup =', soup)

    # Removing current file processed 
    files       = soup.find('Files')
    second_file = files.find_all('File')[0]
    second_file.decompose()
    # print('second_file =', second_file)

    # -1 num of files.
    num_of_files     = soup.find('NumOfFiles')
    num_of_files_val = soup.find('NumOfFiles').text.strip()
    num_of_files_int = int(num_of_files_val)
    num_of_files_int -= 1
    num_of_files.string = str(num_of_files_int)
    # print(num_of_files)
    # print('soup =', soup.prettify())
    
    # saving again to xml file.
    with open(xml_file, 'w') as file_list:
        file_list.write(str(soup.prettify()))
    return num_of_files_int

def writing_again(SA_save_folder=r'C:\Users\ossam\OneDrive\Desktop\SA', file_name_only='C018302_006'):
    # First, it checks if there was a detection in a current detection.
    # if this detection is not a first detetion, in other words, if there was a previous detection and suddenly it stopped for any reason, but the video is not finished yet,
    # we run the detection for this video again. 
    # If the event happened already, the second detection doesn't do anything, but the event doesn't happened yet, it overwrits xml file.
    # This funtion checks if (1) this is the first detection and (2) there was an event in the first detection.
    
    xml_file  = file_name_only + '.xml'
    full_file = os.path.join(SA_save_folder, xml_file)
    print(full_file)
    
    # checking if there is a xml file, coresponding to file_name_only.
    if os.path.exists(full_file):
        print("file_exists! So, it is detect_again!")
        with open(full_file, 'r') as f:
            data = f.read()
        
        bs_data = BeautifulSoup(data, 'xml')
        # print(bs_data)
        
        # checking if there was an event.
        alarm_description = bs_data.find('AlarmDescription')
        # print(alarm_description)
        if alarm_description:
            print(alarm_description.text, "Already exists. No overwriting.")
            return False
        else:
            print('But, no detection. So, write again.')
            return True
    else:
        print("file does not exist. So, this is the first detection.")
        return True
    

def extract_folder_name(path):
    # print(os.listdir(path))

    # print(name for name in os.listdir(path) if os.path.isdir(name))
    
    for name in os.listdir(path):
        if os.path.isdir(name):
            print(name)

path = r'C:\Users\ossam\OneDrive\Desktop\detect'
extract_folder_name(path)
    
# raw_txt_file = r'D:\safety_2022\KISA_certification\KISA_programmes\list_file\file_list.txt'
# file_list(raw_txt_file)


        # print(point)
        # point[0] and point[1] are each vertice coordinate of rectangle.
    #     if not is_point_inside_polygon(point[0], point[1], polygon): 
    #         return False
    # return True

# Example usage:
# rectangle = [(1, 1), (1, 3), (3, 3), (3, 1)]
# polygon   = [(0, 0), (2, 0), (3, 1), (2, 2), (0, 2)]
# # polygon = [(0, 0), (3, 0), (4, 1), (4, 4), (0, 4)]
# print(center_in_poly(rectangle, polygon))


# if is_rectangle_inside_pentagon(rectangle, polygon):
#     print("Rectangle is inside the pentagon.")
# else:
#     print("Rectangle is not inside the pentagon.")

# p1 = Point(24.82,  60.24)
# p2 = Point(24.895, 60.05)

# print(p1)
# video_file = 'C001101_002.mp4'
# file_names = reading_file_list(xml_file=r'C:\Users\ossam\OneDrive\Desktop\file_list.xml')
# # print(len(file_names))

# # loop for each mp4 file.
# for file_name in file_names:
#     print(file_name, "-> Begin!")
#     load_map(file_name)
# video_file = 'C086105_002.mp4'
# # video_file = 'C086200_002.mp4'
# print(load_map(video_file))
#%%
# dq_frame = deque(maxlen=10) # if it returens same seconds for 10 sec. then I will consider current video is finished.
# print(dq_frame)
# dq_frame.append(34)
# print(dq_frame)
# dq_frame.append(34)
# print(dq_frame)
# dq_frame.append(34)
# print(dq_frame)
# dq_frame.append(34)
# print(dq_frame)
# dq_frame.append(34)
# print(dq_frame)
# dq_frame.append(34)
# print(dq_frame)
# dq_frame.append(34)
# print(dq_frame)
# dq_frame.append(34)
# print(dq_frame)
# dq_frame.append(34)
# print(dq_frame)
# dq_frame.append(34)
# print(dq_frame)
# dq_frame.append(34)
# print(dq_frame)
# dq_frame.append(34)
# print(dq_frame)
# dq_frame.append(34)
# print(dq_frame)
# dq_frame.append(34)
# print(dq_frame)
# dq_frame.append(34)
# print(dq_frame)

# all_same = are_all_same(dq_frame)
# print(all_same)

# #%%
# dq = [deque(maxlen=10) for _ in range(3)] # 20 when fps = 30
# print(dq)
# # dq[0] = deque('love')
# dq[0].append(1)
# dq[0].append(1)
# dq[0].append(None)
# dq[0].append(1)

# # # dq[0].append(['2'])
# # # dq[0].append(['3'])
# # # dq[0].append(['4'])
# # # dq[0].append(['5'])
# # # dq[0].append(['6'])
# print(dq[0].count(1))

# print(dq)

#%%
# dq1 = deque((1,2,3), 3)
# print(dq1)

# dq1.append(4)
# print(dq1)


#%%
# Intrusion = True
# intrusion_happen = False
# for i in range(2):
#     print('i =', i)
        
#     if Intrusion and not intrusion_happen:
#         print('Intrusion        =', Intrusion)
#         print('intrusion_happen =', intrusion_happen)
#         intrusion_happen = True
#%%
# ss = "0:{}:{}".format(3,4)
# print(ss)

# xml_writer()





# # map modifier
# file_name = r'C081100_002.mp4'
# img = r'D:\data\runs\detect\C081100_002\C081100_002_4m_3s_243.012_detected.jpg'
# im0 = cv2.imread(img)
# DetectionAreas, DetectArea, Intrusion, Loitering, DetectArea_tup, Intrusion_tup, Loitering_tup = load_map(file_name)
# Drawing_boundaries(im0, DetectionAreas, DetectArea, Loitering, Intrusion)
# cv2.imshow('ddd', im0)
# cv2.waitKey(0) 
# cv2.destroyAllWindows() 
