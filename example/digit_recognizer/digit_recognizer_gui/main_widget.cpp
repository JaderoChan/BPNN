#include "main_widget.h"

#include <cstring>

#include <qstyle.h>
#include <qlabel.h>
#include <qlayout.h>
#include <qfiledialog.h>
#include <qmessagebox.h>

MainWidget::MainWidget(QWidget* parent)
    : QWidget(parent)
{
    setWindowTitle("BPNN Digit Recognizer");

    netParamsFilepathInputer_.setPlaceholderText("Input the net params file path.");
    selectNetParamsFileButton_.setIcon(style()->standardIcon(QStyle::SP_FileIcon));
    selectNetParamsFileButton_.setToolTip("Select a net params file.");
    loadNetParamsButton_.setText("Load net params");
    clearBoardButton_.setText("Clear");
    predict_.setText("Predict");

    board_.setBoardSize(QSize(128, 128));
    board_.setFixedSize(512, 512);
    board_.fill(0);
    board_.setBrushColor(128);
    board_.setBrushSize(8);

    QWidget*     w1 = new QWidget(this);
    w1->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Fixed);
    QHBoxLayout* l1 = new QHBoxLayout(w1);
    l1->setContentsMargins(0, 0, 0, 0);
    l1->addWidget(new QLabel("Net Params:", this));
    l1->addWidget(&netParamsFilepathInputer_);
    l1->addWidget(&selectNetParamsFileButton_);
    l1->addWidget(&loadNetParamsButton_);

    QWidget*     w2 = new QWidget(this);
    QVBoxLayout* l2 = new QVBoxLayout(w2);
    l2->setContentsMargins(QMargins(0, 0, 0, 0));
    l2->addWidget(&board_);
    QWidget*     w3 = new QWidget(w2);
    w3->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Fixed);
    QHBoxLayout* l3 = new QHBoxLayout(w3);
    l3->setContentsMargins(QMargins(0, 0, 0, 0));
    l3->addStretch();
    l3->addWidget(&clearBoardButton_);
    l2->addWidget(w3);

    QVBoxLayout* l0 = new QVBoxLayout(this);
    l0->addWidget(w1);
    l0->addWidget(w2);
    l0->addWidget(&predict_);

    connect(&selectNetParamsFileButton_, &QToolButton::clicked, this, [=]()
    {
        const QString filepath =
            QFileDialog::getOpenFileName(this, "Open the net params file", QDir::currentPath());
        if (!filepath.isEmpty())
            netParamsFilepathInputer_.setText(filepath);
    });

    connect(&loadNetParamsButton_, &QPushButton::clicked, this, [=]()
    {
        const std::string filepath = netParamsFilepathInputer_.text().toStdString();
        const bool ok = bpnn_params_load_from_file(&netParams_, filepath.c_str());
        if (!ok)
            QMessageBox::warning(this, "Warning", "Failed to load the net params file.");
    });

    connect(&clearBoardButton_, &QPushButton::clicked, &board_, [=]()
    { board_.fill(0); });

    connect(&predict_, &QPushButton::clicked, this, &MainWidget::predict);
}

void MainWidget::predict()
{
    if (!bpnn_params_valid(&netParams_))
    {
        QMessageBox::warning(this, "Warning", "The invalid bpnn params.");
        return;
    }

    const QImage src = board_.exportImage();

    // 找到图像有效区域边界（具有笔迹的地方）
    int xMin = src.width(), xMax = -1, yMin = src.height(), yMax = -1;
    for (int y = 0; y < src.height(); ++y)
    {
        const uchar* line = src.constScanLine(y);
        for (int x = 0; x < src.width(); ++x)
        {
            if (line[x] > 35)
            {
                xMin = qMin(xMin, x);
                xMax = qMax(xMax, x);
                yMin = qMin(yMin, y);
                yMax = qMax(yMax, y);
            }
        }
    }

    double ins[28 * 28] = {0.0};

    // 处理图像：居中有效区域，增加边距以模拟 MNIST 的数据集图像。
    if (xMax >= xMin && yMax >= yMin)
    {
        const int bw         = xMax - xMin + 1;
        const int bh         = yMax - yMin + 1;
        const int side       = qMax(bw, bh);
        const int margin     = qMax(2, side / 7);
        const int squareSide = side + 2 * margin;

        const int  cx       = (xMin + xMax) / 2;
        const int  cy       = (yMin + yMax) / 2;
        const QRect cropRect(cx - squareSide / 2, cy - squareSide / 2, squareSide, squareSide);

        QImage cropCanvas(squareSide, squareSide, QImage::Format_Grayscale8);
        cropCanvas.fill(0);
        const QRect validCrop = cropRect.intersected(src.rect());
        if (!validCrop.isEmpty())
        {
            const int dstX = validCrop.x() - cropRect.x();
            const int dstY = validCrop.y() - cropRect.y();
            for (int dy = 0; dy < validCrop.height(); ++dy)
            {
                const uchar* srcLine = src.constScanLine(validCrop.y() + dy);
                uchar*       dstLine = cropCanvas.scanLine(dstY + dy);
                memcpy(dstLine + dstX, srcLine + validCrop.x(), (size_t) validCrop.width());
            }
        }

        const QImage scaled20 = cropCanvas
            .scaled(20, 20, Qt::IgnoreAspectRatio, Qt::SmoothTransformation)
            .convertToFormat(QImage::Format_Grayscale8);
        QImage canvas28(28, 28, QImage::Format_Grayscale8);
        canvas28.fill(0);
        for (int dy = 0; dy < 20; ++dy)
        {
            const uchar* srcLine = scaled20.constScanLine(dy);
            uchar*       dstLine = canvas28.scanLine(4 + dy);
            memcpy(dstLine + 4, srcLine, 20);
        }

        // 归一化数据
        for (int y = 0; y < 28; ++y)
            for (int x = 0; x < 28; ++x)
                ins[y * 28 + x] = (double) canvas28.constScanLine(y)[x] / 255.0;
    }

    double outs[10] = {0.0};
    bpnn_predict(&netParams_, ins, outs);

    int    num = 0;
    double max = 0.0;
    for (int i = 0; i < 10; ++i)
    {
        if (outs[i] > max)
        {
            max = outs[i];
            num = i;
        }
    }

    QString text = QString(
        "Result: %1\n\n"
        "0: %2\n1: %3\n2: %4\n3: %5\n4: %6\n"
        "5: %7\n6: %8\n7: %9\n8: %10\n9: %11\n")
        .arg(num)
        .arg(outs[0], 0, 'f', 6).arg(outs[1], 0, 'f', 6).arg(outs[2], 0, 'f', 6)
        .arg(outs[3], 0, 'f', 6).arg(outs[4], 0, 'f', 6).arg(outs[5], 0, 'f', 6)
        .arg(outs[6], 0, 'f', 6).arg(outs[7], 0, 'f', 6).arg(outs[8], 0, 'f', 6)
        .arg(outs[9], 0, 'f', 6);
    QMessageBox::information(this, "Predict result", text);
}
