#include "pixel_drawing_board.h"

#include <qassert.h>

PixelDrawingBoard::PixelDrawingBoard(QWidget* parent)
    : QWidget(parent)
{
    setMouseTracking(true);
}

PixelDrawingBoard::PixelDrawingBoard(const QImage& image, QWidget* parent)
    : PixelDrawingBoard(parent)
{
    importImage(image, false);
}

PixelDrawingBoard::PixelDrawingBoard(const QSize& boardSize, uchar fillColor, QWidget* parent)
    : PixelDrawingBoard(parent)
{
    importImage(QImage(boardSize, QImage::Format_Grayscale8), false);
    fill(fillColor);
}

PixelDrawingBoard::PixelDrawingBoard(int width, int height, uchar fillColor, QWidget* parent)
    : PixelDrawingBoard(QSize(width, height), fillColor, parent)
{}

QImage PixelDrawingBoard::exportImage() const
{
    return image_;
}

void PixelDrawingBoard::importImage(const QImage& image, bool keepGridSize)
{
    QImage img = image.convertToFormat(QImage::Format_Grayscale8);
    if (keepGridSize)
        img = img.scaled(image_.size());
    image_ = img;
    update();
    emit drawingChanged();
}

void PixelDrawingBoard::fill(uchar color)
{
    image_.fill(QColor(color, color, color));
    update();
    emit drawingChanged();
}

QSize PixelDrawingBoard::boardSize() const
{
    return image_.size();
}

qreal PixelDrawingBoard::gridLineWidth() const
{
    return gridLineWidth_;
}

QColor PixelDrawingBoard::gridLineColor() const
{
    return gridLineColor_;
}

int PixelDrawingBoard::brushSize() const
{
    return brushSize_;
}

int PixelDrawingBoard::brushDelta() const
{
    return brushDelta_;
}

uchar PixelDrawingBoard::pixel(int x, int y) const
{
    Q_ASSERT(x >= 0 && x < image_.width() && y >= 0 && y < image_.height());
    return image_.constScanLine(y)[x];
}

void PixelDrawingBoard::setBoardSize(const QSize& size, uchar fillColor)
{
    image_ = QImage(size, QImage::Format_Grayscale8);
    fill(fillColor);
    updatePixelSize();
    if (mouseInside_)
        updateBrushPoints();
    update();
    emit drawingChanged();
}

void PixelDrawingBoard::setBoardSize(int width, int height, uchar fillColor)
{
    setBoardSize(QSize(width, height), fillColor);
}

void PixelDrawingBoard::setGridLineWidth(qreal width)
{
    const qreal clamped = qMax(0.0, width);
    if (gridLineWidth_ != clamped)
    {
        gridLineWidth_ = clamped;
        update();
    }
}

void PixelDrawingBoard::setGridLineColor(const QColor& color)
{
    if (gridLineColor_ != color)
    {
        gridLineColor_ = color;
        update();
    }
}

void PixelDrawingBoard::setBrushSize(int size)
{
    const int clamped = qMax(0, size);
    if (brushSize_ != clamped)
    {
        brushSize_ = clamped;
        if (mouseInside_)
        {
            updateBrushPoints();
            update();
        }
    }
}

void PixelDrawingBoard::setBrushDelta(int delta)
{
    const int clamped = std::clamp(delta, -UCHAR_MAX, UCHAR_MAX);
    if (brushDelta_ != clamped)
    {
        brushDelta_ = clamped;
        update();
    }
}

void PixelDrawingBoard::setPixel(int x, int y, uchar color)
{
    Q_ASSERT(x >= 0 && x < image_.width() && y >= 0 && y < image_.height());
    uchar& p = image_.scanLine(y)[x];
    if (p != color)
    {
        p = color;
        update();
        emit drawingChanged();
    }
}

QRectF PixelDrawingBoard::pixelRect(int x, int y) const
{
    Q_ASSERT(x >= 0 && x < image_.width() && y >= 0 && y < image_.height());
    return QRectF(
        static_cast<qreal>(x) * pixelSize_.width(),
        static_cast<qreal>(y) * pixelSize_.height(),
        pixelSize_.width(),
        pixelSize_.height());
}

void PixelDrawingBoard::enterEvent(QEnterEvent* event)
{
    QWidget::enterEvent(event);
    cursorPos_   = event->position();
    mouseInside_ = true;
    updateBrushPoints();
    update();
}

void PixelDrawingBoard::leaveEvent(QEvent* event)
{
    QWidget::leaveEvent(event);
    mouseInside_ = false;
    brushPoints_.clear();
    update();
}

void PixelDrawingBoard::mouseMoveEvent(QMouseEvent* event)
{
    cursorPos_ = event->position();
    updateBrushPoints();
    if (event->buttons() & Qt::LeftButton)
        applyBrushDelta();
    update();
}

void PixelDrawingBoard::mousePressEvent(QMouseEvent* event)
{
    if (event->button() == Qt::LeftButton)
    {
        cursorPos_ = event->position();
        updateBrushPoints();
        applyBrushDelta();
        update();
    }
    QWidget::mousePressEvent(event);
}

void PixelDrawingBoard::wheelEvent(QWheelEvent* event)
{
    if (event->modifiers() & Qt::ControlModifier)
    {
        const int delta = event->angleDelta().y() > 0 ? 1 : -1;
        setBrushSize(brushSize_ + delta);
        event->accept();
    }
    else
    {
        QWidget::wheelEvent(event);
    }
}

void PixelDrawingBoard::resizeEvent(QResizeEvent* event)
{
    QWidget::resizeEvent(event);
    updatePixelSize();
    if (mouseInside_)
        updateBrushPoints();
    update();
}

void PixelDrawingBoard::paintEvent(QPaintEvent* event)
{
    QPainter painter(this);

    // 绘制像素
    for (int y = 0; y < image_.height(); ++y)
    {
        for (int x = 0; x < image_.width(); ++x)
        {
            const uchar v = pixel(x, y);
            painter.fillRect(pixelRect(x, y), QColor(v, v, v));
        }
    }

    // 绘制网格边框线
    if (gridLineWidth_ > 0.0)
    {
        painter.setPen(QPen(QBrush(gridLineColor_), gridLineWidth_));

        for (int y = 0; y <= image_.height(); ++y)
        {
            const qreal lineY = static_cast<qreal>(y) * pixelSize_.height();
            painter.drawLine(QPointF(0.0, lineY), QPointF(static_cast<qreal>(width()), lineY));
        }

        for (int x = 0; x <= image_.width(); ++x)
        {
            const qreal lineX = static_cast<qreal>(x) * pixelSize_.width();
            painter.drawLine(QPointF(lineX, 0.0), QPointF(lineX, static_cast<qreal>(height())));
        }
    }

    // 绘制笔刷预览
    if (mouseInside_)
    {
        for (const QPoint& pt : brushPoints_)
        {
            const uchar v = static_cast<uchar>(
                std::clamp(static_cast<int>(pixel(pt.x(), pt.y())) + brushDelta_, 0, UCHAR_MAX));
            painter.fillRect(pixelRect(pt.x(), pt.y()), QColor(v, v, v));
        }
    }
}

void PixelDrawingBoard::updatePixelSize()
{
    if (image_.width() == 0 || image_.height() == 0)
    {
        pixelSize_ = QSizeF();
        return;
    }

    pixelSize_.setWidth(static_cast<qreal>(width()) / static_cast<qreal>(image_.width()));
    pixelSize_.setHeight(static_cast<qreal>(height()) / static_cast<qreal>(image_.height()));
}

void PixelDrawingBoard::updateBrushPoints()
{
    brushPoints_.clear();
    if (pixelSize_.width() == 0.0 || pixelSize_.height() == 0.0)
        return;

    const int px     = static_cast<int>(cursorPos_.x() / pixelSize_.width());
    const int py     = static_cast<int>(cursorPos_.y() / pixelSize_.height());
    const int xStart = qMax(px - brushSize_, 0);
    const int yStart = qMax(py - brushSize_, 0);
    const int xEnd   = qMin(px + brushSize_ + 1, image_.width());
    const int yEnd   = qMin(py + brushSize_ + 1, image_.height());
    for (int y = yStart; y < yEnd; ++y)
    {
        for (int x = xStart; x < xEnd; ++x)
        {
            const qreal dist = std::sqrt(
                static_cast<qreal>((px - x) * (px - x) + (py - y) * (py - y)));
            if (dist < static_cast<qreal>(brushSize_))
                brushPoints_.append(QPoint(x, y));
        }
    }
}

void PixelDrawingBoard::applyBrushDelta()
{
    bool hasPixelChanged = false;
    for (const QPoint& pt : brushPoints_)
    {
        Q_ASSERT(pt.x() >= 0 && pt.x() < image_.width() && pt.y() >= 0 && pt.y() < image_.height());
        uchar* line  = image_.scanLine(pt.y());
        const uchar v = static_cast<uchar>(
            std::clamp(static_cast<int>(line[pt.x()]) + brushDelta_, 0, UCHAR_MAX));
        if (line[pt.x()] != v)
        {
            hasPixelChanged = true;
            line[pt.x()] = v;
        }
    }
    if (hasPixelChanged)
        emit drawingChanged();
}
