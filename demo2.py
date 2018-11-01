import numpy as np
import os
import tensorflow as tf
import random
from multiprocessing import Process, Queue
import copy
import math
from time import time as timer
from time import sleep
import time
import cv2
import argparse
import os
import pprint
import tensorflow as tf
from model import VDSR
import matplotlib.pyplot as plt
import numpy as np
from imresize import *
from utils import *

#======= Select mode ==============
RESULT_FILENAME = "FREE"
#RESULT_FILENAME = "RESULT"
#RESULT_FILENAME = "RESULT2"
#RESULT_FILENAME = "RESULT_static_opt"
#RESULT_FILENAME = "RESULT_static_slow"
#RESULT_FILENAME = "RESULT_static_fast"
#=========================

#======global variables =====================
stress =0
currentOption = 5# meaning start mode
useScenario = False


TIMECONSTRAINT = 0.6 if RESULT_FILENAME != "RESULT2" else 0.3
if RESULT_FILENAME == "RESULT" or RESULT_FILENAME ==  "RESULT2" : currentOption = 0
elif RESULT_FILENAME == "RESULT_static_opt" : currentOption = 4
elif RESULT_FILENAME == "RESULT_static_slow" : currentOption = 3
elif RESULT_FILENAME == "RESULT_static_fast" : currentOption = 5
elif RESULT_FILENAME == "FREE" : stress = 0 ; currentOption = 0 ; useScenario = False

IMAGE_SIZE = (680, 400) #resolution
FP_loc = (20,50)
SR_loc = (20,IMAGE_SIZE[1]+60)
US_loc = (60+500,20)

FRAMERATE = 20 #it is just constant
STARTTIME = time.time() #program start time
#===========================================


#========help codes================================
def load_image_into_numpy_array(image):
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)
#================================================


#=============define help classes=====================
class FrameAndStamps :
    def __init__(self,frame_ , framenumber_,timestamp_, roi,downScaler=1):
        self.framenumber = framenumber_
        self.frame = frame_
        self.timestamp = timestamp_
        self.roi = roi
        self.downScaler = downScaler

    def getTimestamp(self):
        return self.timestamp

    def getFrame(self):
        return self.frame

    def getFramenumber(self):
        return self.framenumber

    def getROI(self):
        return self.roi

    def getDownScaler(self):
        return self.downScaler
#==========================================================




'''*********************************************************************************************************************************************************************************
 * def processFunc_framePutter(frameQueue, controlQueue):
*********************************************************************************************************************************************************************************'''
def processFunc_framePutter(frameQ, processControlQ,framerateControlQ,useScenario):

    #=========================================================================================================
    STARTTIME = time.time()
    framerate = FRAMERATE
    roi_size = 100
    roi_size_max = 500
    roi_x = 10
    roi_y = 10
    input_downscale = 1
    original = False
    camera_framePutter = cv2.VideoCapture(0)
    camera_framePutter.set(3, IMAGE_SIZE[0])
    camera_framePutter.set(4, IMAGE_SIZE[1])
    _, frame = camera_framePutter.read()
    inputrate = 0
    lastAcceptedTimeChecker = 0
    isScenario = useScenario
    #myscenario = scenario.Scenario(STARTTIME)
    cv2.namedWindow('FramePutter', 0)
    cv2.resizeWindow('FramePutter', IMAGE_SIZE[0], IMAGE_SIZE[1])
    cv2.moveWindow("FramePutter",FP_loc[0],FP_loc[0])
    #=========================================================================================================



    # ===========opencv trackbar controllers==================================================================
    def framerateController(x):
        nonlocal framerate
        nonlocal framerateControlQ
        framerate = x
        framerateControlQ.put(x)

    def roiSizeController(x):
        nonlocal roi_size
        roi_size = max(10,x)

    def roixLocationController(x):
        nonlocal roi_x
        roi_x = x

    def roiyLocationController(x):
        nonlocal roi_y
        roi_y = x

    def inputdownscaleController(x):
        nonlocal input_downscale
        input_downscale = max(0.1,x/100)


    def originalController(x):
        nonlocal original
        original = x

    #----------------------------------------------------------------------
    cv2.createTrackbar('original', 'FramePutter', 0, 1, originalController)
    cv2.createTrackbar('frame rate', 'FramePutter', 10, 50, framerateController)
    cv2.createTrackbar('input_downscale', 'FramePutter', 100, 100, inputdownscaleController)
    cv2.createTrackbar('ROI size', 'FramePutter', 10, roi_size_max, roiSizeController)
    cv2.createTrackbar('ROI x', 'FramePutter', 10, IMAGE_SIZE[0]-roi_size, roixLocationController)
    cv2.createTrackbar('ROI y', 'FramePutter', 10, IMAGE_SIZE[1]-roi_size, roiyLocationController)
    #=========================================================================================================




    # ========================================================================================================
    while processControlQ.qsize() == 0 and (isScenario is False or myscenario.isEnd() is False):
        '''
        # ===== expieriment scenario ======================================
        if(isScenario is True):
            myscenario.playScenario()
            if myscenario.getCurrentFramerate() != framerate :
                framerate = myscenario.getCurrentFramerate()

        # =================================================================
        '''

        #----------------------------------------------------------------------
        period = 1 if framerate == 0 else 1.0/framerate #framerate 0 means waiting 1sec
        if time.time() - lastAcceptedTimeChecker >= period :

            # take a photo. then stamp useful infomations.
            # ----------------------------------------------------------------------
            #_ , frame = camera_framePutter.read()
            #frame = np.array(frame[...,::-1])

            image_path = os.path.join(os.getcwd(),"test", "Custom", "monarch.bmp") #urban 9 urban3.png ppt3.bmp
            frame = plt.imread(image_path)

            if np.max(frame) >10 : frame = (frame/ 255).astype(np.float32)
            ds = 1 if original else input_downscale

            frame = imresize(frame, ds)
            roi_set = {'roi_size': int(roi_size*ds), 'roi_x': int(roi_x*ds), 'roi_y': int(roi_y*ds)}


            timestamp = time.time()
            frame_copied = copy.deepcopy(frame)
            framenumber = 0
            frameAndStamps = FrameAndStamps(frame_copied, framenumber, timestamp, roi_set,ds)

            # ----------------------------------------------------------------------
            # coping frame. then put into waitQ
            # ----------------------------------------------------------------------
            frameQ.put(frameAndStamps)
            inputrate = 1.0 / (time.time() - lastAcceptedTimeChecker)
            framerateControlQ.put(inputrate)
            # stamp additional informations on currently input frame.
            # ----------------------------------------------------------------------
            '''
            #cv2.putText(frame, "Frame number : {} ".format(framenumber),   (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3)
            cv2.putText(frame, "Current Input Rate: {} Q len: {}".format(round(1.0/(time.time() - lastAcceptedTimeChecker), 4), frameQ.qsize()),
                        (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3)

            cv2.putText(frame, "total elapsed time : {} sec".format(round(time.time() - STARTTIME), 4),
                        (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3)

            '''
            cv2.rectangle(frame, (int(roi_x*ds), int(roi_y*ds)), (int((roi_x+roi_size)*ds), int((roi_y+roi_size)*ds)), (0, 255, 0), 3)
            cv2.imshow('FramePutter', frame[...,::-1])
            # ----------------------------------------------------------------------

            lastAcceptedTimeChecker = time.time()
            pass

        else :
            sleep(0.00001)
            pass
        # ----------------------------------------------------------------------


        # quit the program on the press of key 'w'
        #----------------------------------------------------------------------
        if cv2.waitKey(1) & 0xFF == ord('w'):
            break
        #----------------------------------------------------------------------

    #===========  end loop  =================================================================================




    # == clean up process ================================================================================
    camera_framePutter.release()
    cv2.destroyAllWindows()
    # ==================================================================================================

'''*********************************************************************************************************************************************************************************
 * end function
*********************************************************************************************************************************************************************************'''





#=================load graph and lables==========================================
#==========================================================================








'''*********************************************************************************************************************************************************************************
 * MAIN PROCEDURE
*********************************************************************************************************************************************************************************'''
if __name__ == "__main__" :
# ==========================================
# model configuration
# ==========================================
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_tag", type=str, default="Depth Adaptable SR net demo")
    parser.add_argument("--gpu", type=int, default=0) # -1 for CPU
    parser.add_argument("--patch_size", type=int, default=41)
    parser.add_argument("--num_additional_inputlayer", type=int, default=0)
    parser.add_argument("--num_hiddens",type=int, default=7)
    parser.add_argument("--num_innerlayer", type=int, default=3)
    parser.add_argument("--num_additional_branchlayer", type=int, default=0)
    parser.add_argument("--scale", type=int, default=3) #
    parser.add_argument("--checkpoint_dir", default="checkpoint")
    parser.add_argument("--cpkt_itr", default=40)  # -1 for latest, set 0 for training from scratch
    parser.add_argument("--result_dir", default="result")
    parser.add_argument("--infer_subdir", default="Custom")
    parser.add_argument("--infer_imgpath", default="webtoon2.bmp") #monarch.bmp #webtoon1.png #ppt3.bmp #lenna.bmp
    parser.add_argument("--type", default="demo", choices=["eval", "demo"]) #eval type uses .m data upscaled with matlab bicubic mathod on only Y chaanel, demo type uses raw images on RGB
    parser.add_argument("--mode", default="demo", choices=["train", "test", "inference", "test_plot", "demo"])
    parser.add_argument("--c_dim", type=int, default=-1)  # 3 for RGB, 1 for Y chaanel of YCbCr
    parser.add_argument("--save_extension", default=".jpg", choices=["jpg", "png"])

    print("=====================================================================")
    args = parser.parse_args()
    if args.type == "eval" : args.c_dim = 1 ; args.train_subdir += "_M" ; args.test_subdir += "_M"
    elif args.type == "demo" : args.c_dim = 1 ;
    print("Eaxperiment tag : " + args.exp_tag)
    print(args)
    print("=====================================================================")

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    config = tf.ConfigProto()
    if args.gpu == -1: config.device_count = {'GPU': 0}
    config.gpu_options.per_process_gpu_memory_fraction = 0.9
    # config.operation_timeout_in_ms=10000

    # -----------------------------------
    # build sess
    # -----------------------------------
    g = tf.Graph()
    g.as_default()
    sess = tf.Session(config=config)

    # -----------------------------------
    # build model
    # -----------------------------------
    model_path = args.checkpoint_dir
    vdsr = VDSR(sess, args=args)




# ==========================================
# system configuration
# ==========================================
    frameQ = Queue()
    processControlQ = Queue()
    framerateControlQ = Queue()
    process_framePutter = Process(target=processFunc_framePutter, args=(frameQ, processControlQ, framerateControlQ,useScenario ))
    process_framePutter.start()

    processingTime = 0.1
    ffelapsedTime = 0.05 #feed forwarding time
    current_model = 0 #depth of layer
    lastChanged = 0

    f = open(RESULT_FILENAME, 'w')
    f.close()
    f = open(RESULT_FILENAME, 'a')

    STARTTIME = time.time()
    cv2.namedWindow('super resolution', 0)
    cv2.resizeWindow('super resolution', 500, 50)
    cv2.moveWindow("super resolution", SR_loc[0],SR_loc[1])



# ==========================================
# variables for lyapunov optimization
# ==========================================
    timelimit = TIMECONSTRAINT #sec
    QMax = 20
    framerate = 15
    V = 0.01
    expOutputRate =0
    expInputRate = 0



# ==========================================
# variables for evaluation
# ==========================================
    fps = 0
    cumulatedTimeaveragePerformance = 0
    timestamp = 0
    delay = 0

    bufferlen_hist = []
    outputRate_hist = []
    model_hist = []
    delay_hist=[]
    plt.ion()
    plt.show
    draw_timechecker = 0
    append_timechecker = 0


# ==========================================
# opencv trackbar controllers
# ===========================================
    def stressController(x):
        global stress
        stress = x
        pass


    def optionController(x):
        global currentOption
        currentOption = x
        pass


    def scaleController(x):
        global args
        args.scale = x/100+0.1
        pass

    cv2.createTrackbar('stress','super resolution', 0, 255, stressController)
    cv2.createTrackbar('scale', 'super resolution', 0, 400, scaleController)
    cv2.createTrackbar('currentOptions', 'super resolution', 0, args.num_hiddens+2, optionController) #0 for auto, last for nonprocess


    '''
    def calDrift(M,M_CUR,QSIZE) :
        drift = 0

        # ========cal variables related with model i ==================
        # model i's and model last's expected ouput rate
        nonfftime = processingTime - ffelapsedTime

        expPTime_M = ( nonfftime + ffelapsedTime * (M['speed'] / M_CUR['speed']))  # expected processing time when using model M
        expOutputRate_M = 1.0 / expPTime_M  # expected outputRate when using model i
        expExhautionRate_M = (1 - (framerate - expOutputRate_M) / expOutputRate_M)
        # ===============================================================


        #=======cal Qmax=============
        QMax = int(timelimit/(1/expOutputRate_M))
        if(QSIZE == -1) : QSIZE = QMax
        #============================


        # ========cal drift =========================
        rewardTerm = M['mAP'] * min(framerate if not framerate == 0 else 0.00001 ,expOutputRate_M ) if expOutputRate_M> (1/timelimit) else 0
        backlogTerm = QSIZE * expOutputRate_M

        drift = V * rewardTerm  + backlogTerm
        drift_dict = {'drift' : drift, 'rewardTerm':rewardTerm, 'backlogTerm':backlogTerm}
        #===================================

        return drift_dict
    #====================================
    '''

# ============================================================
# main loop
# ============================================================
    while True:
        processingTimeChecker = time.time()  # -- timer
        # ---------------------------------------------------------------
        # load fram from buffer
        # ---------------------------------------------------------------
        if(framerateControlQ.qsize()>0) :
            while (framerateControlQ.qsize() > 1):
                if cv2.waitKey(1) & 0xFF == ord('q') or (process_framePutter.is_alive() == False and frameQ.qsize() <= 0):
                    processControlQ.put(1)
                    break
                framerateControlQ.get()
            framerate = framerateControlQ.get()

        frameAndStamps = copy.deepcopy(frameQ.get())
        args.scale = 1/frameAndStamps.getDownScaler()
        image_np = frameAndStamps.getFrame()
        if np.max(image_np) > 1: image_np = image_np / 255
        roi = frameAndStamps.getROI()
        image_np = image_np[roi['roi_y'] : roi['roi_y']+ roi['roi_size'] , roi['roi_x'] : roi['roi_x']+ roi['roi_size']]
        # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
        image_np_expanded = np.expand_dims(image_np, axis=0)



        # ============================== model selection  =======================================================================
        # =====================auto adapting mode ================================================
        if(currentOption == 0) :
            pass
            '''
            if(time.time() - lastChanged > 0.5): # timelimit/2) : #adaptation period
  
                #=========================== cal V ===========================================
                VTEMP = 9999
                temp = 0
  
                MODEL_last = MODEL_LIST[len(MODEL_LIST)-1]
                MODEL_last_last = MODEL_LIST[len(MODEL_LIST)-2]
  
                #========cal min VTEMP ====
                calDrift_I = calDrift(MODEL_last_last,MODEL,-1)
                calDrift_Last = calDrift(MODEL_last,MODEL,-1)
  
                temp = (calDrift_Last['backlogTerm'] - calDrift_I['backlogTerm']) / (calDrift_I['rewardTerm'] - calDrift_Last['rewardTerm']) *0.7
                if temp < VTEMP and temp >=0:
                    VTEMP = temp
                #=========================================================================
                pass
  
                V = VTEMP if VTEMP >0 and VTEMP<9999 else V
  
                #============================ cal V end  ====================================
  
  
  
  
                #============================ select optimal model ==============================
                MODELTEMP = MODEL_LIST[len(MODEL_LIST)-1]              #init model with fastest but having lowest performance to handle too fast sampling rate or too heavy stress
                maxTemp = -9999
                temp = 0
                for i in MODEL_LIST:
  
                    #========cal variables related with model i       ==================
                    #model i's expected ouput rate
                    nonfftime = processingTime - ffelapsedTime
                    expPTime_i =  (nonfftime + ffelapsedTime * (i['speed'] / MODEL['speed']))    #expected processing time when using model i
                    expOutputRate_i = 1.0 / expPTime_i                                       #expected outputRate when using model i
                    # ========cal variables related with model i end  ==================
  
  
                    #===============cal max drift ====================================
                    #temp = V * i's mAP  * ( 1 - (framerate- outputrate_i) / outputrate_i  ) ^ (qsize/qMax)  + qsize * outpurate_i
                    #temp = V * i['mAP'] * pow((1 - (framerate - expOutputRate_i) / expOutputRate_i), (frameQ.qsize() / QMax)) + frameQ.qsize()* expOutputRate_i
                    temp = calDrift(i,MODEL,frameQ.qsize())['drift']
                    if temp > maxTemp:
                        MODELTEMP = i
                        expOutputRate = expOutputRate_i
                        expInputRate = framerate
                        maxTemp = temp
                    #===========================================================
                    pass
                #============================ select optimal model end ==============================
  
  
  
  
                #============================ adapt model ==============================
                MODEL = MODELTEMP
                lastChange = time.time()
                #============================ adapt model end ==========================
            pass
        # =====================auto adapting mode ================================================
           '''



      # ===================== static mode ================================================
        elif currentOption < args.num_hiddens+3 :
            current_model = currentOption-1
            expOutputRate = 1/processingTime
            expInputRate = framerate

        else : #currentOption == len(MODEL_LIST)+1   #do not predict. this means a dummy model whose speed is max and accurate is 0.
            pass
      # ===================== static mode end  ===========================================
      # ============================== model selection end =============================================================






        # ============================== super resolution  =============================================================
        processingTimeChecker = time.time()  # -- timer
        ffelapsedTimeChecker = time.time()


        if current_model == 0:
            print("free")
            pass

        elif current_model == 1:
            image_np = imresize(image_np, scalar_scale=args.scale, output_shape=None)
            #image_np = imresize(image_np, scalar_scale=1/frameAndStamps.getDownScaler(), output_shape=None)


        else :
            image_np =  ycbcr2rgb(vdsr.inference(rgb2ycbcr(image_np), depth = current_model-2, scale = args.scale))
            #image_np = vdsr.inference(image_np, depth = current_model-2, scale = 1/frameAndStamps.getDownScaler())

        # ============================== super resolution  end ========================================================
        ffelapsedTimeTemp = time.time() - ffelapsedTimeChecker
        ffelapsedTime = ffelapsedTimeTemp
        #ffelapsedTime = 0.5 * ffelapsedTimeTemp + 0.5 * ffelapsedTime



        # ==== wait as stress ========================================================================================
        # simulate stress. Now, it is just arbitrary wait function
        mywait = random.uniform(stress - stress * 0, stress + stress * 0)
        mywait = mywait if (mywait > 0) else 0
        sleep(mywait / 122)
        # ==== wait as stress end ====================================================================================

        processingTimeTemp = time.time() - processingTimeChecker
        processingTime = processingTimeTemp
        #processingTime = 0.5*processingTimeTemp + 0.5* processingTime






        # ================== post process     ==========================================================================
        #================ cal result ========================
        #fps
        fps = 1.0 / processingTime

        #current model
        currentmodel = str("NONE") if currentOption == args.num_hiddens+3 else str(current_model)
        currentmodelNumber = str("NONE") if currentOption == args.num_hiddens+3 else str(current_model)

        #delay
        delay = time.time() - frameAndStamps.getTimestamp()

        #time stamp
        timestamp = time.time() - STARTTIME

        #time average performance
        cumulatedTimeaveragePerformance += 1 #MODEL['mAP']
        timeaveragePerformance = cumulatedTimeaveragePerformance / (time.time() - STARTTIME)

        #print log
        print(round(timestamp,3),"\t", currentmodel ,"\t",round(fps,3),"\t",round(expOutputRate,3),"\t",round(delay,3),"\t",round(timeaveragePerformance,3))
        data = \
          str(round(timestamp,3))+ "\t" + \
          currentmodelNumber + "\t"+ \
          str(round(fps,3))+ "\t" +\
          str(round(expOutputRate,3))+ "\t" + \
          str(round(delay,3))+ "\t" + \
          str(round(timeaveragePerformance,3)) + "\t"+ \
          str(round(timelimit,3)) + "\t" +\
          str(framerate) + "\n"


        f.write(data)
        #================ cal result end  ====================



        #===    post useful describtion  ======================
        #cv2.putText(image_np, "Current Model:" + currentmodel                                                     , (20,  50) , cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3)
        #cv2.putText(image_np, "Current Output Rate: {} /sec ".format(round(fps, 4))                               , (10, 20) , cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
        #cv2.putText(image_np, "Expected Input Rate: {} /sec ".format(round(expInputRate, 4))                      , (20, 150) , cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3)
        #cv2.putText(image_np, "Expected Output Rate: {} /sec ".format(round(expOutputRate, 4))                    , (20, 200) , cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3)

        #cv2.putText(image_np, "Q len: {}".format(frameQ.qsize())                                                  , (10, 40) , cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
        #cv2.putText(image_np, "Delay: {}".format(round(delay,4))                                                  , (10, 60) , cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
        #cv2.putText(image_np, "ffelapsedTime: {}".format(round(ffelapsedTime,4))                                  , (20, 350) , cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3)

        #cv2.putText(image_np, "time average performance: {}".format(round(timeaveragePerformance,4))              , (20, 400) , cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3)

        cv2.imshow('upscaled image', cv2.resize(image_np[...,::-1], (int(roi['roi_size']*args.scale), int(roi['roi_size']*args.scale))))
        '''
        if time.time() - append_timechecker > 0.1 :
            temp = time.time()
            outputRate_hist.append(round(fps, 4))
            bufferlen_hist.append(frameQ.qsize())
            delay_hist.append(delay)
            model_hist.append(current_model)
            append_timechecker = time.time()

        if time.time() - draw_timechecker > 5:
            temp = time.time()
            plt.clf()
            outputrate_plot = plt.plot(outputRate_hist,label = "outputRate")
            bufferlen_plot = plt.plot(bufferlen_hist,label = "bufferlen_plot")
            delay_plot = plt.plot(delay_hist,label = 'delay_hist')
            model_plot = plt.plot(model_hist, label = 'model_hist')
            plt.legend()#handles = [outputrate_plot, bufferlen_plot, delay_plot, model_plot])
            plt.draw()
            plt.pause(0.001)
            print("draw" , time.time() - temp)
            draw_timechecker = time.time()
        '''
        # =================================================
        #================== post process end   ==========================================================================




        #===cv exit handler========
        if cv2.waitKey(1) & 0xFF == ord('q') or (process_framePutter.is_alive() == False and frameQ.qsize() <= 0):
            processControlQ.put(1)
            break
        #===========================
# =================   main loop end   =========================================


#=============clean up ===========================================
f.close()
cv2.destroyAllWindows()
process_framePutter.join()
#=======================================================================
#===================================================================================

'''*********************************************************************************************************************************************************************************
 * END MAIN
*********************************************************************************************************************************************************************************'''

