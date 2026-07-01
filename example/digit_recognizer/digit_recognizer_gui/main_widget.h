#pragma once

#include <qlineedit.h>
#include <qpushbutton.h>
#include <qslider.h>
#include <qtoolbutton.h>
#include <qwidget.h>

#include <bpnn.h>
#include "color_dot_indicator.h"
#include "pixel_drawing_board.h"

class MainWidget : public QWidget
{
public:
    explicit MainWidget(QWidget* parent = nullptr);

    void predict();

private:
    QLineEdit           modelPathInputer_;
    QToolButton         browseModelButton_;
    QPushButton         loadModelButton_;
    QSlider             brushDeltaSlider_;
    QPushButton         clearBoardButton_;
    QPushButton         predictButton_;
    ColorDotIndicator   indicator_;
    PixelDrawingBoard   pixelDrawingBoard_;
    bpnn_params_t       netParams_  = BPNN_PARAMS_INIT;
};
