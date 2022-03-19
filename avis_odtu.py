import cv2
import numpy as np
import math
import time
from dronekit import connect, VehicleMode, LocationGlobal, LocationGlobalRelative, Command
from pymavlink import mavutil
import threading

# import pytesseract

#################################### Connection String and vehicle connection
print("Connecting")
connection_string = "/dev/ttyAMA0"
vehicle = connect(connection_string, baud=57600, wait_ready=True)
#vehicle = connect('127.0.0.1:14550', wait_ready=True) #Ardupilot simulation
print("Connected")

##VARIABLES
arrow_info = [[0, 0], [0, 0], [0, 0]]  # [ANGLE][ARROW'S_NUMBER]
ok_sayisi = []
zero_to_nine = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]  # Filter for pyteserract
x, y, z = 0, 0, 0
line_frame_angle = 0
line_angle = 0
yaw_global = 0
clock_wise = 1
errors = []
land_side = 0
was_line = 0
Kp = 0.112
Ki = 0
kd = 1
integral1 = 0
derivative = 0
error_corr = 0
ok_count = 0


##GORUNTU ISLEME SINIFI
class drone_vision():
    def init(self):
        pass

    def line_detect(self, frame):
        global line_frame_angle, line_angle, yaw_global, clock_wise, errors, error_corr, integral1, velocity
        mask = cv2.inRange(frame, (0, 0, 0), (180, 255, 40))
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.erode(mask, kernel, iterations=5)
        mask = cv2.dilate(mask, kernel, iterations=9)
        contours, _ = cv2.findContours(mask.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        x = frame.shape[0]
        y = frame.shape[1]
        d = x // 2
        e = y // 2
        cv2.circle(frame, (e, d), 2, (0, 0, 255), 5)
        print(e, d)
        for cnt in contours:
            approx = cv2.approxPolyDP(cnt, 0.01 * cv2.arcLength(cnt, True), True)
            area = cv2.contourArea(cnt)
            if area > 5000:
                was_line = 1
                cv2.drawContours(frame, [approx], 0, (255, 255, 0), 5)
                blackbox = cv2.minAreaRect(cnt)
                box = cv2.boxPoints(blackbox)
                box = np.int0(box)
                (x_min, y_min), (w_min, h_min), angle = blackbox
                error = int(x_min - e)
                if error > 0:
                    line_side = 1  # line in right
                elif error <= 0:
                    line_side = -1  # line in left
                normal_error = float(error) / e
                integral = float(integral1 + normal_error)
                last_error = normal_error
                derivative = normal_error - last_error
                error_corr = -1 * (Kp * normal_error + Ki * integral + kd * derivative)  # PID controler
                print(error_corr)
                cv2.drawContours(frame, [box], 0, (0, 0, 255), 3)
                rectangle_X_point = 0
                rectangle_Y_point = 0
                for i in range(4):
                    rectangle_X_point += box[i][0]
                    rectangle_Y_point += box[i][1]
                atan2 = math.atan2(int(rectangle_Y_point / 4) - d, (int(rectangle_X_point / 4) - e))
                line_frame_angle = math.degrees(atan2)
                print("Merkez Noktanin acisi:", line_frame_angle)
                yaw_global = line_frame_angle
                cv2.circle(frame, (int(rectangle_X_point / 4), int(rectangle_Y_point / 4)), 2, (255, 255, 0), 5)
                normal_ang = float(angle) / 90
                if angle < -45:
                    angle = 90 + angle
                if w_min < h_min and angle > 0:
                    angle = (90 - angle) * -1
                if w_min > h_min and angle < 0:
                    angle = 90 + angle
                line_angle = int(angle)
                print("Line egim aci: ", angle)

                cv2.line(frame, (int(rectangle_X_point / 4), int(rectangle_Y_point / 4)), (e, d), (255, 0, 255), 3)
                if line_frame_angle < 0:
                    if line_frame_angle > -180 and line_frame_angle < -90:
                        clock_wise = -1
                        yaw_global = (line_frame_angle + 90.0) + 360.0
                    else:
                        yaw_global = line_frame_angle + 90.0
                        clockwise = 1
                else:
                    yaw_global = (line_frame_angle) + 90.0
                    clock_wise = 1
                cv2.putText(frame, "line_angle: " + str(line_angle),
                            (int(rectangle_X_point / 4) + 10, int(rectangle_Y_point / 4) + 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2,
                            cv2.LINE_AA)
                return line_angle, line_frame_angle, yaw_global, clock_wise, error_corr

    # Kontur kullanarak detect ettigimiz letter detection kodu
    #
    #
    def letter_detection(self, frame):
        self.gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self.blurred = cv2.GaussianBlur(self.gray, (5, 5), 0)
        self.edged = cv2.Canny(self.blurred, 80, 255, 255)
        contours, _ = cv2.findContours(self.edged, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            area = cv2.contourArea(contour)
            approx = cv2.approxPolyDP(contour, 0.01 * cv2.arcLength(contour, True), True)
            x = approx.ravel()[0]
            y = approx.ravel()[1]
            if area > 900:
                if len(approx) == 12:
                    x1 = approx.ravel()[4]
                    y1 = approx.ravel()[5]
                    x_son = int((x + x1) / 2)
                    y_son = int((y + y1) / 2)
                    print(frame[y_son, x_son, 0])
                    if frame[y_son, x_son, 0] > 120 and len(letter_list) != 1:
                        cv2.putText(frame, "X", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255))
                        if letter_list[-1] != 'X':
                            letter_list.append("X")
                    elif (frame[y_son, x_son, 0] == 0 or frame[y_son, x_son, 0] <= 120) and len(letter_list) == 1:
                        print("Black")
                        # letter_list.append("H")
                        cv2.putText(frame, "H", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255))
                elif len(approx) == 8 and letter_list[-1] == "T" and letter_list[-2] == "X" and letter_list[-3] == "L":
                    cv2.putText(frame, "T", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255))
                    if letter_list[-1] != 'T':
                        letter_list.append("T")
                    return "T"
                elif len(approx) >= 6 and len(approx) < 8 and letter_list[-1] == "X" and len(ok_sayisi) == 3:
                    cv2.putText(frame, "L", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255))
                    if letter_list[-1] != 'L':
                        letter_list.append("L")
                    return "L"
        # cv2.imshow("edged", self.edged)

    # OK DETECTION KODU ICIN MASKELEME
    def arrow_preprocess(self, frame):
        self.img_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self.img_blur = cv2.GaussianBlur(self.img_gray, (5, 5), 1)
        self.img_canny = cv2.Canny(self.img_blur, 50, 50)
        self.kernel = np.ones((3, 3))
        self.img_dilate = cv2.dilate(self.img_canny, self.kernel, iterations=2)
        self.img_erode = cv2.erode(self.img_dilate, self.kernel, iterations=1)
        return self.img_erode

    # Ok'un ucunu bulmaya yarayan kod.
    # setdiff1d ile iki arraydeki aynı olmayan noktaları aldık
    def find_tip(self, points, convex_hull):
        self.length = len(points)
        self.indices = np.setdiff1d(range(self.length), convex_hull)
        for i in range(2):
            j = self.indices[i] + 2
            if j > self.length - 1:
                j = self.length - j
            if np.all(points[j] == points[self.indices[i - 1] - 2]):
                return tuple(points[j])

    # Ok'u detect eden kod.
    #
    #
    def find_arrow(self, frame):
        global clock_wise, arrow_counter, yaw_global, arrow_info
        contours, _ = cv2.findContours(self.arrow_preprocess(frame), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        for cnt in contours:
            self.area = cv2.contourArea(cnt)
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.025 * peri, True)
            hull = cv2.convexHull(approx, returnPoints=False)
            sides = len(hull)
            if self.area > 900:
                if 6 > sides > 3 and sides + 2 == len(approx):
                    if 7 <= len(approx) < 10:
                        for i in approx[0]:
                            a0 = i[0]
                            b0 = i[1]
                            cv2.circle(frame, (int(a0), int(b0)), 2, (0, 0, 255), 2)
                        for i in approx[1]:
                            a1 = i[0]
                            b1 = i[1]
                            cv2.circle(frame, (int(a1), int(b1)), 2, (0, 0, 255), 2)
                        for i in approx[2]:
                            a2 = i[0]
                            b2 = i[1]
                            cv2.circle(frame, (int(a2), int(b2)), 2, (0, 0, 255), 2)
                        for i in approx[3]:
                            a3 = i[0]
                            b3 = i[1]
                            cv2.circle(frame, (int(a3), int(b3)), 2, (0, 0, 255), 2)
                        for i in approx[4]:
                            a4 = i[0]
                            b4 = i[1]
                            cv2.circle(frame, (int(a4), int(b4)), 2, (0, 0, 255), 2)
                        for i in approx[5]:
                            a5 = i[0]
                            b5 = i[1]
                            cv2.circle(frame, (int(a5), int(b5)), 2, (0, 0, 255), 2)
                        for i in approx[6]:
                            a6 = i[0]
                            b6 = i[1]
                            cv2.circle(frame, (int(a6), int(b6)), 2, (0, 0, 255), 2)
                        am = (a0 + a1 + a2 + a3 + a4 + a5 + a6) / 7
                        bm = (b0 + b1 + b2 + b3 + b4 + b5 + b6) / 7

                        cv2.circle(frame, (int(am), int(bm)), 2, (0, 255, 0), 2)
                        arrow_tip = self.find_tip(approx[:, 0, :], hull.squeeze())

                        if arrow_tip:
                            cv2.drawContours(frame, [cnt], -1, (0, 255, 0), 3)
                            cv2.circle(frame, arrow_tip, 3, (0, 0, 255), cv2.FILLED)
                            print(arrow_tip)
                            atan = math.atan2(arrow_tip[1] - bm, arrow_tip[0] - am)
                            self.angle = math.degrees(atan)
                            print(self.angle)
                            yaw_global = self.angle
                            if self.angle < 0:
                                if self.angle > -180 and self.angle < -90:
                                    clock_wise = -1
                                    yaw_global = yaw_global = (self.angle + 90.0) + 360.0
                                else:
                                    yaw_global = self.angle + 90.0
                                    clockwise = 1
                            else:
                                clock_wise = 1

                            cv2.putText(frame, "->" + " Angle: " + "{:.2f}".format(yaw_global), (30, 400),
                                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1,
                                        cv2.LINE_AA)

                            return yaw_global, clock_wise


class drone_fly():
    def init(self, ):
        pass

    # X,y,z eksenlerinde velocity ile hareketi sagladigimiz kod.
    def send_global_velocity(self, velocity_x, velocity_y, velocity_z, duration):
        global error_corr, velocity, altitute
        if vehicle.armed == True:
            while True:
                msg = vehicle.message_factory.set_position_target_global_int_encode(
                    0,  # time_boot_ms (not used)
                    0, 0,  # target system, target component
                    mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT_INT,  # frame
                    0b0000111111000111,  # type_mask (only speeds enabled)
                    0,  # lat_int - X Position in WGS84 frame in 1e7 * meters
                    0,  # lon_int - Y Position in WGS84 frame in 1e7 * meters
                    0,  # alt - Altitude in meters in AMSL altitude(not WGS84 if absolute or relative)
                    # altitude above terrain if GLOBAL_TERRAIN_ALT_INT
                    velocity_x,  # X velocity in NED frame in m/s
                    -error_corr,  # Y velocity in NED frame in m/s
                    velocity_z,  # Z velocity in NED frame in m/s
                    0, 0, 0,  # afx, afy, afz acceleration (not supported yet, ignored in GCS_Mavlink)
                    0, 0)  # yaw, yaw_rate (not supported yet, ignored in GCS_Mavlink)

                # send command to vehicle on 1 Hz cycle
                for x in range(0, duration):
                    vehicle.send_mavlink(msg)
                    time.sleep(1)
        else:
            print("Waiting for Armed")

    # kanatlari calistiran ve dron'u verdigimiz yukseklige yukselten kod
    def arm_and_takeoff(self, aTargetAltitude):
        print("Basic pre-arm checks")
        while not (vehicle.is_armable):
            print("Waiting for vehicle to initialise...")
            time.sleep(1)
        print("Arming motors")
        vehicle.mode = VehicleMode("GUIDED")
        vehicle.armed = True
        while not (vehicle.armed):
            print("Waiting for arming...")
            time.sleep(1)
        print("Taking off!")
        vehicle.simple_takeoff(aTargetAltitude)
        while (True):
            print("Altitude: ", vehicle.location.global_relative_frame.alt)
            if (vehicle.location.global_relative_frame.alt >= aTargetAltitude * 0.95):
                print("Reached target altitude")
            break
        time.sleep(1)

    # inis kodu
    def landing(self):
        print("LAND Mode Active!Pls wait")
        vehicle.mode = VehicleMode("LAND")
        while (vehicle.armed):
            print("Waiting for landing and disarm!")
            time.sleep(1)
        print("Altitude: ", vehicle.location.global_relative_frame.alt)
        print("Vehicle has landed.")

    # dron'u aciyla dondurmemizi saglayan kod.
    def condition_yaw(self, relative=False):
        global line_frame_angle, yaw_global, clock_wise
        yaw_speed = 25  # param 2, yaw speed deg/s
        direction = 1  # 1 for clockwise -1 for reverse
        if relative:
            is_relative = 1  # yaw relative to direction of travel
        else:
            is_relative = 0  # yaw is an absolute angle
            # create the CONDITION_YAW command using command_long_encode()
            msg = vehicle.message_factory.command_long_encode(
                0, 0,  # target system, target component
                mavutil.mavlink.MAV_CMD_CONDITION_YAW,  # command
                0,  # confirmation
                yaw_global,  # param 1, yaw in degrees
                yaw_speed,  # param 2, yaw speed deg/s
                0,  # param 3, direction -1 ccw, 1 cw
                is_relative,  # param 4, relative offset 1, absolute angle 0
                0, 0, 0)  # param 5 ~ 7 not used
            # send command to vehicle
            vehicle.send_mavlink(msg)
            print("Yaw", yaw_global)
            print("CLock Wise", clock_wise)

    #####! Ucus kodlarimiz icin thread
    def startThread(self):
        print("Start Thread")
        t = threading.Thread(target=self.send_global_velocity(0.3, error_corr, 0, 1))
        t.daemon = True
        t.start()


# Sınıflardan obje olusturma kodlari
fly_obj = drone_fly()
drone_obj = drone_vision()
letter_list = ['', "H"]


# Goruntu isleme Kodlari ve Ucus kodlarindan gelen outputlara gore izlenen main code
def main_code():
    global ok_count
    cap = cv2.VideoCapture(0)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('avis_odtu.avi', fourcc, 20.0, (640, 480))
    while True:
        _, frame = cap.read()
        drone_obj.letter_detection(frame)
        if letter_list[-1] == "H":
            print("H default detected")
            print("Line Following mode activated")
            cv2.putText(frame, "Mode:Line Takip Modu", (30, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2,
                        cv2.LINE_AA)
            drone_obj.line_detect(frame)
            fly_obj.condition_yaw()
        elif letter_list[-1] == "X":
            error_corr = 0
            drone_obj.find_arrow(frame)
            fly_obj.condition_yaw()
            cv2.putText(frame, "Mode:Ok Takip Modu", (30, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2,
                        cv2.LINE_AA)
        elif letter_list[-1] == "L":
            cv2.putText(frame, "Mode:Line Takip Modu", (30, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2,
                        cv2.LINE_AA)
            drone_obj.line_detect(frame)
        elif letter_list[-1] == "T":
            cv2.putText(frame, "Mode:Land Mode", (30, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2,
                        cv2.LINE_AA)
            fly_obj.landing()
        cv2.putText(frame, "Son Okunan Harf: " + letter_list[-1], (30, 450),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2,
                    cv2.LINE_AA)
        print("Letter List", letter_list)
        out.write(frame)
        cv2.imshow("Frame", frame)
        k = cv2.waitKey(1)
        if k == 27:
            break

    cv2.destroyAllWindows()


# Main kodun threadi
def th_main():
    th = threading.Thread(target=main_code)
    th.daemon = True
    th.start()


fly_obj.arm_and_takeoff(2.2)  # 2.2 metre yuksel
time.sleep(5)
th_main()  # Main code thread'i
fly_obj.startThread()  # Ucus kodu thread'i

"""
#####! PYTESERRACT FPS DUSUSUNE SEBEBİYET VERDIGI YUZDEN KODDAN CIKARTILMISTIR.
def listToString(ok_sayisi):
   str1 = ""
   for i in ok_sayisi:
      str1 += i
   return str1


def pyt_thread():
    th=threading.Thread(target=pytesseractt)
    th.daemon=True
    th.start()



    print(listToString(ok_sayisi))
    cv2.putText(frame, str(listToString(ok_sayisi)), (0, 500), cv2.FONT_HERSHEY_SIMPLEX, 5, (0, 0, 0), 2)
    cv2.imshow("screen_noise", mask)
    cv2.imshow("img", frame)
"""