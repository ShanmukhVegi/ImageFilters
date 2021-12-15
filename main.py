from Eye_Filter import Eye_Filter
from Dog_Filter import Dog_Filter
from BW_Filter import BW_Filter
from Mustache_Filter import Mustache_Filter

while(True):
    print("Please enter\n1. Eye Filter\n2. Dog filter\n3. Mustache Filter\n4. Black And White Filter\nAny other key to exit")
    k=input()
    if(k=="1"):
        eye_filter = Eye_Filter.Eye_Filter()
        eye_filter.start_process()
    elif(k=="2"):
        dog_filter = Dog_Filter.Dog_Filter()
        dog_filter.start_process()
    elif(k=="3"):
        mustache_filter= Mustache_Filter.Mustache_Filter()
        mustache_filter.start_process()
    elif(k=="4"):
        bw_filter = BW_Filter.BW_Filter()
        bw_filter.start_process()
    else:
        print("EXIT")
        break