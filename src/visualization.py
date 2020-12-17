import pygame as pg
import numpy as np
import os


class InitializeScreen:
    def __init__(self, screenWidth, screenHeight, fullScreen):
        self.screenWidth = screenWidth
        self.screenHeight = screenHeight
        self.fullScreen = fullScreen

    def __call__(self):
        pg.init()
        if self.fullScreen:
            screen = pg.display.set_mode((self.screenWidth, self.screenHeight), pg.FULLSCREEN)
        else:
            screen = pg.display.set_mode((self.screenWidth, self.screenHeight))
        pg.display.init()
        pg.fastevent.init()
        return screen


class DrawBackground():
    def __init__(self, screen, gridSize, leaveEdgeSpace, backgroundColor, lineColor, lineWidth, textColorTuple):
        self.screen = screen
        self.gridSize = gridSize
        self.leaveEdgeSpace = leaveEdgeSpace
        self.widthLineStepSpace = np.int(screen.get_width() / (gridSize + 2 * self.leaveEdgeSpace))
        self.heightLineStepSpace = np.int(screen.get_height() / (gridSize + 2 * self.leaveEdgeSpace))
        self.backgroundColor = backgroundColor
        self.lineColor = lineColor
        self.lineWidth = lineWidth
        self.textColorTuple = textColorTuple

    def __call__(self):
        self.screen.fill((0, 0, 0))
        pg.draw.rect(self.screen, self.backgroundColor, pg.Rect(np.int(self.leaveEdgeSpace * self.widthLineStepSpace), np.int(self.leaveEdgeSpace * self.heightLineStepSpace),
                                                                np.int(self.gridSize * self.widthLineStepSpace), np.int(self.gridSize * self.heightLineStepSpace)))
        for i in range(self.gridSize + 1):
            pg.draw.line(self.screen, self.lineColor, [np.int((i + self.leaveEdgeSpace) * self.widthLineStepSpace), np.int(self.leaveEdgeSpace * self.heightLineStepSpace)],
                         [np.int((i + self.leaveEdgeSpace) * self.widthLineStepSpace), np.int((self.gridSize + self.leaveEdgeSpace) * self.heightLineStepSpace)], self.lineWidth)
            pg.draw.line(self.screen, self.lineColor, [np.int(self.leaveEdgeSpace * self.widthLineStepSpace), np.int((i + self.leaveEdgeSpace) * self.heightLineStepSpace)],
                         [np.int((self.gridSize + self.leaveEdgeSpace) * self.widthLineStepSpace), np.int((i + self.leaveEdgeSpace) * self.heightLineStepSpace)], self.lineWidth)
        return self.screen


class DrawNewState():
    def __init__(self, screen, drawBackground, targetColor, playerColor, targetRadius, playerRadius):
        self.screen = screen
        self.drawBackground = drawBackground
        self.targetColor = targetColor
        self.playerColor = playerColor
        self.targetRadius = targetRadius
        self.playerRadius = playerRadius
        self.leaveEdgeSpace = drawBackground.leaveEdgeSpace
        self.widthLineStepSpace = drawBackground.widthLineStepSpace
        self.heightLineStepSpace = drawBackground.heightLineStepSpace

    def __call__(self, playerPosition, targetPositions, obstacles):
        for event in pg.event.get():
            if event.type == pg.QUIT:
                pg.quit()
            if event.type == pg.KEYDOWN:
                if event.key == pg.K_ESCAPE:
                    pg.quit()
                    exit()
        self.drawBackground()
        pg.draw.circle(self.screen, self.playerColor, [np.int((playerPosition[0] + self.leaveEdgeSpace + 0.5) * self.widthLineStepSpace), np.int((playerPosition[1] + self.leaveEdgeSpace + 0.5) * self.heightLineStepSpace)], self.playerRadius)

        for targerPosition in targetPositions:
            pg.draw.rect(self.screen, self.targetColor, [np.int((targerPosition[0] + self.leaveEdgeSpace + 0.2) * self.widthLineStepSpace), np.int((targerPosition[1] + self.leaveEdgeSpace + 0.2) * self.heightLineStepSpace), self.targetRadius * 2, self.targetRadius * 2])

        [pg.draw.rect(self.screen, [0, 0, 0], [np.int((obstacle[0] + self.leaveEdgeSpace) * self.widthLineStepSpace), np.int((obstacle[1] + self.leaveEdgeSpace) * self.heightLineStepSpace), self.widthLineStepSpace, self.widthLineStepSpace]) for obstacle in obstacles]

        pg.display.flip()
        return self.screen


class DrawImage():
    def __init__(self, screen):
        self.screen = screen
        self.screenCenter = (self.screen.get_width() / 2, self.screen.get_height() / 2)

    def __call__(self, image):
        imageRect = image.get_rect()
        imageRect.center = self.screenCenter
        pause = True
        pg.event.set_allowed([pg.KEYDOWN, pg.KEYUP, pg.QUIT])
        self.screen.fill((0, 0, 0))
        self.screen.blit(image, imageRect)
        pg.display.flip()
        while pause:
            pg.time.wait(10)
            for event in pg.event.get():
                if event.type == pg.KEYDOWN and event.key == pg.K_SPACE:
                    pause = False
                elif event.type == pg.QUIT:
                    pg.quit()
            pg.time.wait(10)
        pg.event.set_blocked([pg.KEYDOWN, pg.KEYUP, pg.QUIT])


class DrawText():
    def __init__(self, screen, drawBackground):
        self.screen = screen
        self.screenCenter = (self.screen.get_width() / 2, self.screen.get_height() / 2)
        self.drawBackground = drawBackground
        self.leaveEdgeSpace = drawBackground.leaveEdgeSpace
        self.widthLineStepSpace = drawBackground.widthLineStepSpace
        self.heightLineStepSpace = drawBackground.heightLineStepSpace

    def __call__(self, text, textColorTuple, textPositionTuple):
        self.drawBackground()
        font = pg.font.Font(None, 50)
        textObj = font.render(text, 1, textColorTuple)
        self.screen.blit(textObj, [np.int((textPositionTuple[0] + self.leaveEdgeSpace + 0.2) * self.widthLineStepSpace),
                                   np.int((textPositionTuple[1] + self.leaveEdgeSpace - 0.1) * self.heightLineStepSpace)])
        pg.display.flip()
        return


if __name__ == "__main__":
    gridSize = [15, 15]
    screenWidth = 600
    screenHeight = 600
    fullScreen = False

    initializeScreen = InitializeScreen(screenWidth, screenHeight, fullScreen)
    screen = initializeScreen()
    # pg.mouse.set_visible(False)

    leaveEdgeSpace = 2
    lineWidth = 1
    backgroundColor = [205, 255, 204]
    lineColor = [0, 0, 0]
    targetColor = [255, 50, 50]
    playerColor = [50, 50, 255]
    targetRadius = 10
    playerRadius = 10
    textColorTuple = (255, 50, 50)

    drawBackground = DrawBackground(screen, gridSize[0], leaveEdgeSpace, backgroundColor, lineColor, lineWidth, textColorTuple)
    drawNewState = DrawNewState(screen, drawBackground, targetColor, playerColor, targetRadius, playerRadius)
    playerPosition = [10, 10]
    targetPositions = [(1, 2), (3, 4), (5, 6)]
    obstacles = []

    for i in range(10):
        drawNewState(playerPosition, targetPositions, obstacles)
    pg.time.wait(500)
