import cv2
import mediapipe as mp
import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *

import math
import time

pygame.init()

screen = (1400, 900)

pygame.display.set_mode(screen, DOUBLEBUF | OPENGL)

pygame.display.set_caption("Gesture Hologram")

gluPerspective(45, screen[0] / screen[1], 0.1, 50.0)

glEnable(GL_DEPTH_TEST)

cam = cv2.VideoCapture(0)

mpHands = mp.solutions.hands

hands = mpHands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

rotX = 0
rotY = 0

zoom = -8

objIndex = 0

lastSwipe = time.time()

autoRotate = False

rotateSpeed = 0.5

names = [
    "Cube",
    "Pyramid",
    "Sphere",
    "Diamond",
    "Cylinder",
    "Double Pyramid",
    "Wire Sphere",
    "Wire Cube"
]

def text(x, y, msg):

    font = pygame.font.SysFont("Arial", 24)

    surface = font.render(msg, True, (0, 255, 255))

    data = pygame.image.tostring(surface, "RGBA", True)

    glWindowPos2d(x, y)

    glDrawPixels(
        surface.get_width(),
        surface.get_height(),
        GL_RGBA,
        GL_UNSIGNED_BYTE,
        data
    )

def cube():

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

def pyramid():

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

def sphere():

    glColor3f(0, 1, 0)

    q = gluNewQuadric()

    gluSphere(q, 1, 40, 40)

def diamond():

    glColor3f(1, 1, 0)

    points = [
        (0,1,0),
        (1,0,0),
        (0,-1,0),
        (-1,0,0),
        (0,0,1),
        (0,0,-1)
    ]

    links = [
        (0,1),(0,3),(0,4),(0,5),
        (2,1),(2,3),(2,4),(2,5)
    ]

    glBegin(GL_LINES)

    for l in links:
        for v in l:
            glVertex3fv(points[v])

    glEnd()

def cylinder():

    glColor3f(0, 0.5, 1)

    q = gluNewQuadric()

    gluCylinder(q, 1, 1, 2, 32, 32)

def doublePyramid():

    glColor3f(1, 0, 0)

    points = [
        (0,1,0),
        (-1,0,1),
        (1,0,1),
        (1,0,-1),
        (-1,0,-1),
        (0,-1,0)
    ]

    lines = [
        (0,1),(0,2),(0,3),(0,4),
        (5,1),(5,2),(5,3),(5,4),
        (1,2),(2,3),(3,4),(4,1)
    ]

    glBegin(GL_LINES)

    for line in lines:
        for v in line:
            glVertex3fv(points[v])

    glEnd()

def wireSphere():

    glColor3f(1, 0.5, 0)

    q = gluNewQuadric()

    gluQuadricDrawStyle(q, GLU_LINE)

    gluSphere(q, 1, 20, 20)

def wireCube():

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

def drawObj():

    if objIndex == 0:
        cube()

    elif objIndex == 1:
        pyramid()

    elif objIndex == 2:
        sphere()

    elif objIndex == 3:
        diamond()

    elif objIndex == 4:
        cylinder()

    elif objIndex == 5:
        doublePyramid()

    elif objIndex == 6:
        wireSphere()

    elif objIndex == 7:
        wireCube()

while True:

    ok, frame = cam.read()

    if not ok:
        break

    frame = cv2.flip(frame, 1)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    result = hands.process(rgb)

    if result.multi_hand_landmarks:

        for hand in result.multi_hand_landmarks:

            ix = hand.landmark[8].x
            iy = hand.landmark[8].y

            tx = hand.landmark[4].x
            ty = hand.landmark[4].y

            wx = hand.landmark[0].x

            rotX = iy * 360
            rotY = ix * 360

            dist = math.sqrt(
                (tx - ix) ** 2 +
                (ty - iy) ** 2
            )

            zoom = -3 - (dist * 25)

            move = ix - wx

            now = time.time()

            if move > 0.20 and now - lastSwipe > 2:

                objIndex += 1

                objIndex %= 8

                lastSwipe = now

                print("next object")

            elif move < -0.20 and now - lastSwipe > 2:

                objIndex -= 1

                objIndex %= 8

                lastSwipe = now

                print("previous object")

            # open hand detection

            finger1 = hand.landmark[8].y < hand.landmark[6].y
            finger2 = hand.landmark[12].y < hand.landmark[10].y
            finger3 = hand.landmark[16].y < hand.landmark[14].y
            finger4 = hand.landmark[20].y < hand.landmark[18].y

            if finger1 and finger2 and finger3 and finger4:

                autoRotate = True

            else:

                autoRotate = False

            mp.solutions.drawing_utils.draw_landmarks(
                frame,
                hand,
                mpHands.HAND_CONNECTIONS
            )

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

    glLoadIdentity()

    gluPerspective(45, screen[0] / screen[1], 0.1, 50.0)

    glTranslatef(0.0, 0.0, zoom)

    if autoRotate:
        rotY += rotateSpeed

    glRotatef(rotX, 1, 0, 0)
    glRotatef(rotY, 0, 1, 0)

    drawObj()

    text(20, 850, f"Object : {names[objIndex]}")
    text(20, 810, "Move Hand = Rotate")
    text(20, 770, "Pinch = Zoom")
    text(20, 730, "Swipe = Change Object")
    text(20, 690, "Open Palm = Auto Rotate")

    pygame.display.flip()

    pygame.time.wait(10)

    cv2.putText(
        frame,
        "Gesture Hologram",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0,255,255),
        2
    )

    cv2.imshow("Hand Tracking", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam.release()

cv2.destroyAllWindows()

pygame.quit()
