import cv2
import mediapipe as mp
import pygame
from pygame.locals import *

from OpenGL.GL import *
from OpenGL.GLU import *

import math
import time

# =========================================
# MEDIAPIPE SETUP
# =========================================

mp_hands = mp.solutions.hands

hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

# =========================================
# PYGAME + OPENGL
# =========================================

pygame.init()

display = (1400, 900)

pygame.display.set_mode(display, DOUBLEBUF | OPENGL)

pygame.display.set_caption("Gesture Hologram AI PRO")

gluPerspective(45, (display[0] / display[1]), 0.1, 50.0)

glEnable(GL_DEPTH_TEST)

# =========================================
# CAMERA
# =========================================

cap = cv2.VideoCapture(0)

# =========================================
# VARIABLES
# =========================================

rotation_x = 0
rotation_y = 0

zoom = -8

current_object = 0

last_swipe_time = time.time()

auto_rotate = False

rotation_speed = 0.5

paused = False

# =========================================
# OBJECT NAMES
# =========================================

object_names = [
    "CUBE",
    "PYRAMID",
    "SPHERE",
    "DIAMOND",
    "CYLINDER",
    "DOUBLE PYRAMID",
    "WIRE SPHERE",
    "WIRE CUBE"
]

# =========================================
# DRAW TEXT
# =========================================

def draw_text(x, y, text):

    font = pygame.font.SysFont("Arial", 26, True)

    text_surface = font.render(text, True, (0, 255, 255))

    text_data = pygame.image.tostring(
        text_surface,
        "RGBA",
        True
    )

    glWindowPos2d(x, y)

    glDrawPixels(
        text_surface.get_width(),
        text_surface.get_height(),
        GL_RGBA,
        GL_UNSIGNED_BYTE,
        text_data
    )

# =========================================
# OBJECTS
# =========================================

def draw_cube():

    glColor3f(0, 1, 1)

    vertices = (
        (1,-1,-1),
        (1,1,-1),
        (-1,1,-1),
        (-1,-1,-1),
        (1,-1,1),
        (1,1,1),
        (-1,-1,1),
        (-1,1,1)
    )

    edges = (
        (0,1),(0,3),(0,4),
        (2,1),(2,3),(2,7),
        (6,3),(6,4),(6,7),
        (5,1),(5,4),(5,7)
    )

    glBegin(GL_LINES)

    for edge in edges:
        for vertex in edge:
            glVertex3fv(vertices[vertex])

    glEnd()

# =========================================

def draw_pyramid():

    glColor3f(1, 0, 1)

    vertices = [
        (0,1,0),
        (-1,-1,1),
        (1,-1,1),
        (1,-1,-1),
        (-1,-1,-1)
    ]

    edges = [
        (0,1),(0,2),(0,3),(0,4),
        (1,2),(2,3),(3,4),(4,1)
    ]

    glBegin(GL_LINES)

    for edge in edges:
        for vertex in edge:
            glVertex3fv(vertices[vertex])

    glEnd()

# =========================================

def draw_sphere():

    glColor3f(0, 1, 0)

    quadric = gluNewQuadric()

    gluSphere(quadric, 1, 40, 40)

# =========================================

def draw_diamond():

    glColor3f(1, 1, 0)

    points = [
        (0,1,0),
        (1,0,0),
        (0,-1,0),
        (-1,0,0),
        (0,0,1),
        (0,0,-1)
    ]

    connections = [
        (0,1),(0,3),(0,4),(0,5),
        (2,1),(2,3),(2,4),(2,5)
    ]

    glBegin(GL_LINES)

    for c in connections:
        for v in c:
            glVertex3fv(points[v])

    glEnd()

# =========================================

def draw_cylinder():

    glColor3f(0, 0.5, 1)

    quadric = gluNewQuadric()

    gluCylinder(quadric, 1, 1, 2, 32, 32)

# =========================================

def draw_double_pyramid():

    glColor3f(1, 0, 0)

    points = [
        (0,1,0),
        (-1,0,1),
        (1,0,1),
        (1,0,-1),
        (-1,0,-1),
        (0,-1,0)
    ]

    connections = [
        (0,1),(0,2),(0,3),(0,4),
        (5,1),(5,2),(5,3),(5,4),
        (1,2),(2,3),(3,4),(4,1)
    ]

    glBegin(GL_LINES)

    for c in connections:
        for v in c:
            glVertex3fv(points[v])

    glEnd()

# =========================================

def draw_wire_sphere():

    glColor3f(1, 0.5, 0)

    quadric = gluNewQuadric()

    gluQuadricDrawStyle(quadric, GLU_LINE)

    gluSphere(quadric, 1, 20, 20)

# =========================================

def draw_wire_cube():

    glColor3f(1, 1, 1)

    vertices = (
        (1,-1,-1),
        (1,1,-1),
        (-1,1,-1),
        (-1,-1,-1),
        (1,-1,1),
        (1,1,1),
        (-1,-1,1),
        (-1,1,1)
    )

    edges = (
        (0,1),(0,3),(0,4),
        (2,1),(2,3),(2,7),
        (6,3),(6,4),(6,7),
        (5,1),(5,4),(5,7)
    )

    glBegin(GL_LINES)

    for edge in edges:
        for vertex in edge:
            glVertex3fv(vertices[vertex])

    glEnd()

# =========================================
# DRAW CURRENT OBJECT
# =========================================

def draw_object():

    if current_object == 0:
        draw_cube()

    elif current_object == 1:
        draw_pyramid()

    elif current_object == 2:
        draw_sphere()

    elif current_object == 3:
        draw_diamond()

    elif current_object == 4:
        draw_cylinder()

    elif current_object == 5:
        draw_double_pyramid()

    elif current_object == 6:
        draw_wire_sphere()

    elif current_object == 7:
        draw_wire_cube()

# =========================================
# MAIN LOOP
# =========================================

while True:

    success, frame = cap.read()

    if not success:
        break

    frame = cv2.flip(frame, 1)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(rgb)

    if results.multi_hand_landmarks:

        for hand_landmarks in results.multi_hand_landmarks:

            # Index Finger
            ix = hand_landmarks.landmark[8].x
            iy = hand_landmarks.landmark[8].y

            # Thumb
            tx = hand_landmarks.landmark[4].x
            ty = hand_landmarks.landmark[4].y

            # Wrist
            wx = hand_landmarks.landmark[0].x

            # =========================================
            # ROTATION
            # =========================================

            if not paused:

                rotation_x = iy * 360
                rotation_y = ix * 360

            # =========================================
            # ZOOM
            # =========================================

            distance = math.sqrt(
                (tx - ix) ** 2 +
                (ty - iy) ** 2
            )

            zoom = -3 - (distance * 25)

            # =========================================
            # SWIPE
            # =========================================

            current_time = time.time()

            movement = ix - wx

            # RIGHT SWIPE
            if movement > 0.20 and current_time - last_swipe_time > 2:

                current_object += 1

                if current_object > 7:
                    current_object = 0

                last_swipe_time = current_time

            # LEFT SWIPE
            elif movement < -0.20 and current_time - last_swipe_time > 2:

                current_object -= 1

                if current_object < 0:
                    current_object = 7

                last_swipe_time = current_time

            # =========================================
            # AUTO ROTATE
            # =========================================

            index_up = hand_landmarks.landmark[8].y
            middle_up = hand_landmarks.landmark[12].y
            ring_up = hand_landmarks.landmark[16].y
            pinky_up = hand_landmarks.landmark[20].y

            if (
                index_up < hand_landmarks.landmark[6].y and
                middle_up < hand_landmarks.landmark[10].y and
                ring_up < hand_landmarks.landmark[14].y and
                pinky_up < hand_landmarks.landmark[18].y
            ):

                auto_rotate = True

            else:

                auto_rotate = False

            # =========================================
            # DRAW LANDMARKS
            # =========================================

            mp.solutions.drawing_utils.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )

    # =========================================
    # OPENGL
    # =========================================

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

    glLoadIdentity()

    gluPerspective(45, (display[0] / display[1]), 0.1, 50.0)

    glTranslatef(0.0, 0.0, zoom)

    # AUTO ROTATE
    if auto_rotate:

        rotation_y += rotation_speed

    glRotatef(rotation_x, 1, 0, 0)

    glRotatef(rotation_y, 0, 1, 0)

    # DRAW OBJECT
    draw_object()

    # =========================================
    # HUD
    # =========================================

    draw_text(20, 850, f"OBJECT : {object_names[current_object]}")

    draw_text(20, 810, "MOVE HAND = ROTATE")

    draw_text(20, 770, "PINCH = ZOOM")

    draw_text(20, 730, "SWIPE = CHANGE OBJECT")

    draw_text(20, 690, "OPEN PALM = AUTO ROTATE")

    pygame.display.flip()

    pygame.time.wait(10)

    # =========================================
    # CAMERA WINDOW
    # =========================================

    cv2.putText(
        frame,
        "Gesture Hologram AI PRO",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0,255,255),
        2
    )

    cv2.imshow("Hand Tracking", frame)

    # =========================================
    # EXIT
    # =========================================

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# =========================================
# CLEANUP
# =========================================

cap.release()

cv2.destroyAllWindows()

pygame.quit()