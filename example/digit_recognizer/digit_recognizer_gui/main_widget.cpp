#include "main_widget.h"

#include <cstring>

#include <qdialog.h>
#include <qfiledialog.h>
#include <qlabel.h>
#include <qlayout.h>
#include <qlist.h>
#include <qmessagebox.h>
#include <qpixmap.h>
#include <qstyle.h>

constexpr QSizePolicy VERTICAL_FIXED_POLICY(QSizePolicy::Expanding, QSizePolicy::Fixed);

class PredictResultDialog : public QDialog
{
public:
    PredictResultDialog(const QImage& inputImage, double* confidences, int result,
        QWidget* parent = nullptr)
        : QDialog(parent)
    {
        setWindowTitle("Result");

        // Image display
        QWidget* imgWgt = new QWidget(this);
        imgWgt->setSizePolicy(VERTICAL_FIXED_POLICY);
        {
            QLabel* imgLabel = new QLabel(imgWgt);
            imgLabel->setPixmap(QPixmap::fromImage(inputImage));
            imgLabel->setFixedSize(56, 56);
            imgLabel->setScaledContents(true);

            QHBoxLayout* l = new QHBoxLayout(imgWgt);
            l->setContentsMargins(0, 0, 0, 0);
            l->addStretch();
            l->addWidget(imgLabel);
            l->addStretch();
        }

        // Result display
        QWidget* resultWgt = new QWidget(this);
        resultWgt->setSizePolicy(VERTICAL_FIXED_POLICY);
        {
            const QString text = QString("Result: %1").arg(result);
            QLabel* resultLabel = new QLabel(text, resultWgt);
            QFont font = resultLabel->font();
            font.setPointSize(13);
            font.setBold(true);
            resultLabel->setFont(font);
            resultLabel->setAlignment(Qt::AlignCenter);

            QHBoxLayout* l = new QHBoxLayout(resultWgt);
            l->setContentsMargins(0, 0, 0, 0);
            l->addWidget(resultLabel);
        }

        // Confidences display
        QWidget* confidencesWgt = new QWidget(this);
        {
            QVBoxLayout* l = new QVBoxLayout(confidencesWgt);
            l->setContentsMargins(0, 0, 0, 0);
            for (int i = 0; i < 10; ++i)
            {
                const QString text = QString("%1: %2").arg(i).arg(confidences[i], 0, 'f', 6);
                QLabel* label = new QLabel(text, confidencesWgt);
                label->setSizePolicy(VERTICAL_FIXED_POLICY);
                label->setAlignment(Qt::AlignCenter);
                l->addWidget(label);
            }
        }

        QVBoxLayout* l = new QVBoxLayout(this);
        l->addWidget(imgWgt);
        l->addWidget(resultWgt);
        l->addWidget(confidencesWgt);
    }
};

MainWidget::MainWidget(QWidget* parent)
    : QWidget(parent)
{
    setWindowTitle("BPNN Digit Recognizer");

    // Top UI
    QWidget* topWgt = new QWidget(this);
    topWgt->setSizePolicy(VERTICAL_FIXED_POLICY);
    {
        indicator_.setSizePolicy(QSizePolicy::Fixed, QSizePolicy::Fixed);
        modelPathInputer_.setPlaceholderText("Input the model file path");
        browseModelButton_.setIcon(style()->standardIcon(QStyle::SP_FileIcon));
        browseModelButton_.setToolTip("Open a model file");
        loadModelButton_.setText("Load");

        QHBoxLayout* l = new QHBoxLayout(topWgt);
        l->setContentsMargins(0, 0, 0, 0);
        l->addWidget(&indicator_);
        l->addWidget(new QLabel("Model:", topWgt));
        l->addWidget(&modelPathInputer_);
        l->addWidget(&browseModelButton_);
        l->addWidget(&loadModelButton_);
    }

    // Pixel drawing board UI
    QWidget* boardWgt = new QWidget(this);
    {
        pixelDrawingBoard_.setBoardSize(128, 128);
        pixelDrawingBoard_.setFixedSize(512, 512);
        pixelDrawingBoard_.fill(0);
        pixelDrawingBoard_.setBrushDelta(128);
        pixelDrawingBoard_.setBrushSize(8);

        QWidget* boardToolWgt = new QWidget(this);
        {
            brushDeltaSlider_.setOrientation(Qt::Horizontal);
            brushDeltaSlider_.setRange(-255, 255);
            brushDeltaSlider_.setValue(128);
            brushDeltaSlider_.setToolTip(QString::number(128));
            clearBoardButton_.setText("Clear");
            clearBoardButton_.setShortcut(QKeySequence(QKeyCombination(Qt::CTRL, Qt::Key_C)));

            QHBoxLayout* l = new QHBoxLayout(boardToolWgt);
            l->setContentsMargins(0, 0, 0, 0);
            l->addWidget(new QLabel("Brush delta:", boardToolWgt));
            l->addWidget(&brushDeltaSlider_);
            l->addWidget(&clearBoardButton_);
        }

        QVBoxLayout* l = new QVBoxLayout(boardWgt);
        l->setContentsMargins(0, 0, 0, 0);
        l->setSpacing(1);
        l->addWidget(&pixelDrawingBoard_);
        l->addWidget(boardToolWgt);
    }

    // Bottom UI
    QWidget* bottomWgt = new QWidget(this);
    bottomWgt->setSizePolicy(VERTICAL_FIXED_POLICY);
    {
        predictButton_.setText("Predict");
        predictButton_.setShortcut(QKeySequence(QKeyCombination(Qt::CTRL, Qt::Key_Return)));

        QHBoxLayout* l = new QHBoxLayout(bottomWgt);
        l->setContentsMargins(0, 0, 0, 0);
        l->addWidget(&predictButton_);
    }

    QVBoxLayout* l = new QVBoxLayout(this);
    l->addWidget(topWgt);
    l->addWidget(boardWgt);
    l->addWidget(bottomWgt);

    // Connects
    connect(&browseModelButton_, &QToolButton::clicked, this, [=]()
    {
        const QString filepath =
            QFileDialog::getOpenFileName(this, "Open a model file", QDir::currentPath());
        if (!filepath.isEmpty())
            modelPathInputer_.setText(filepath);
    });
    connect(&loadModelButton_, &QPushButton::clicked, this, [=]()
    {
        const std::string filepath = modelPathInputer_.text().toStdString();
        const bool ok = bpnn_params_load_from_file(&netParams_, filepath.c_str());
        if (!ok)
            QMessageBox::warning(
                this, "Warning", QString("Failed to load the model file: \"%1\".").arg(filepath));
        else
            indicator_.setColor(Qt::green);
    });

    connect(&brushDeltaSlider_, &QSlider::valueChanged, this, [=](int value)
    {
        brushDeltaSlider_.setToolTip(QString::number(value));
        pixelDrawingBoard_.setBrushDelta(value);
    });
    connect(&clearBoardButton_, &QPushButton::clicked, this, [=]()
    {
        pixelDrawingBoard_.fill(0);
    });

    connect(&predictButton_, &QPushButton::clicked, this, &MainWidget::predict);
}

static QRect boudingOfDrawn(const QImage& image, uchar baseColor)
{
    int xMin = image.width(), xMax = 0, yMin = image.height(), yMax = 0;
    for (int y = 0; y < image.height(); ++y)
    {
        const uchar* line = image.constScanLine(y);
        for (int x = 0; x < image.width(); ++x)
        {
            if (line[x] != baseColor)
            {
                xMin = qMin(xMin, x);
                xMax = qMax(xMax, x);
                yMin = qMin(yMin, y);
                yMax = qMax(yMax, y);
            }
        }
    }

    if (xMax >= xMin && yMax >= yMin)
        return QRect(QPoint(xMin, yMin), QPoint(xMax, yMax));
    return image.rect();
}

void MainWidget::predict()
{
    if (!bpnn_params_valid(&netParams_))
    {
        QMessageBox::warning(this, "Warning", "Current model is invalid.");
        return;
    }

    const QImage src = pixelDrawingBoard_.exportImage();

    // 预处理图像
    // 1. 获取图像有效区域（已绘制区域）包围盒。
    // 2. 构造一个方形图像，其尺寸为有效区域的最小正方形包围盒大小。
    // 3. 将有效区域居中显示在方形图像上。
    // 4. 将方形图像缩放至 20 * 20 大小，再居中显示到 28 * 28 的结果图像上。
    QImage normalizedImg(28, 28, QImage::Format_Grayscale8);
    normalizedImg.fill(QColor(0, 0, 0));
    const QRect drawnRect = boudingOfDrawn(src, 0);
    if (drawnRect.isValid())
    {
        const int squareSide = qMax(drawnRect.width(), drawnRect.height());
        const int xMargin = (squareSide - drawnRect.width())  / 2;
        const int yMargin = (squareSide - drawnRect.height()) / 2;

        QImage canvas(squareSide, squareSide, QImage::Format_Grayscale8);
        canvas.fill(QColor(0, 0, 0));
        for (int y = 0; y < drawnRect.height(); ++y)
        {
            const uchar* srcLine = src.constScanLine(drawnRect.y() + y);
            uchar*       dstLine = canvas.scanLine(yMargin + y);
            memcpy(dstLine + xMargin, srcLine + drawnRect.x(), drawnRect.width());
        }
        canvas = canvas.scaled(20, 20);

        for (int y = 0; y < 20; ++y)
        {
            const uchar* srcLine = canvas.constScanLine(y);
            uchar*       dstLine = normalizedImg.scanLine(y + 4);
            memcpy(dstLine + 4, srcLine, 20);
        }
    }

    // 归一化数据
    double ins[28 * 28] = {0.0};
    for (int y = 0; y < 28; ++y)
        for (int x = 0; x < 28; ++x)
            ins[y * 28 + x] = static_cast<double>(normalizedImg.constScanLine(y)[x]) / 255.0;

    // 推理预测
    double outs[10] = {0.0};
    bpnn_predict(&netParams_, ins, outs);

    // 解析结果
    int result = 0;
    {
        double maxConfidence = -INFINITY;
        for (int i = 0; i < 10; ++i)
        {
            if (outs[i] > maxConfidence)
            {
                maxConfidence = outs[i];
                result = i;
            }
        }
    }

    // 显示结果
    PredictResultDialog dlg(normalizedImg, outs, result, this);
    dlg.exec();
}
