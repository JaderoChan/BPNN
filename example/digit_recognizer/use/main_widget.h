#include <qwidget.h>
#include <qpushbutton.h>
#include <qtoolbutton.h>
#include <qlineedit.h>

#include <bpnn.h>
#include "pixel_drawing_board.h"

class MainWidget : public QWidget
{
public:
    explicit MainWidget(QWidget* parent = nullptr);

    void predict();

private:
    QLineEdit           netParamsFilepathInputer_;
    QToolButton         selectNetParamsFileButton_;
    QPushButton         loadNetParamsButton_;
    QPushButton         clearBoardButton_;
    QPushButton         predict_;
    PixelDrawingBoard   board_;
    bpnn_params_t       netParams_  = BPNN_PARAMS_INIT;
};
