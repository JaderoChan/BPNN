#include "pixel_drawing_board.h"

#include <assert.h>

PixelDrawingBoard::PixelDrawingBoard(QWidget* parent)
    : QWidget(parent), image_(0, 0, QImage::Format_Grayscale8)
{
    setMouseTracking(true);
}

PixelDrawingBoard::PixelDrawingBoard(const QImage& image, QWidget* parent)
    : QWidget(parent)
{
    setMouseTracking(true);
    image_ = image.convertToFormat(QImage::Format_Grayscale8);
}

PixelDrawingBoard::PixelDrawingBoard(const QSize& boardSize, uchar fillColor, QWidget* parent)
    : QWidget(parent), image_(boardSize, QImage::Format_Grayscale8)
{
    setMouseTracking(true);
    fill(fillColor);
}

QImage PixelDrawingBoard::exportImage() const
{
    return image_;
}

void PixelDrawingBoard::importImage(const QImage& image)
{
    image_ = image.convertToFormat(QImage::Format_Grayscale8).scaled(image_.size());
    update();
}

void PixelDrawingBoard::fill(uchar color)
{
    image_.fill(QColor(qRgb(color, color, color)));
    update();
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

int PixelDrawingBoard::brushColor() const
{
    return brushColor_;
}

uchar PixelDrawingBoard::pixel(int x, int y) const
{
    assert(x >= 0 && x < image_.width() && y >= 0 && y < image_.height());
    return image_.constScanLine(y)[x];
}

void PixelDrawingBoard::setBoardSize(const QSize& size, uchar fillColor)
{
    image_ = QImage(size, QImage::Format_Grayscale8);
    fill(fillColor);
    updatePixelSize();
    if (isEntered_) updateBrushPoints();
    update();
}

void PixelDrawingBoard::setGridLineWidth(qreal value)
{
    const qreal nv = (value < 0.0 ? 0.0 : value);
    if (gridLineWidth_ != nv)
    {
        gridLineWidth_ = nv;
        updatePixelSize();
        if (isEntered_) updateBrushPoints();
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

void PixelDrawingBoard::setBrushSize(int value)
{
    const int nv = (value < 0 ? 0 : value);
    if (brushSize_ != nv)
    {
        brushSize_ = nv;
        if (isEntered_)
        {
            updateBrushPoints();
            update();
        }
    }
}

void PixelDrawingBoard::setBrushColor(int value)
{
    const int nv = std::clamp(value, -UCHAR_MAX, UCHAR_MAX);
    if (brushColor_ != nv)
    {
        brushColor_ = nv;
        update();
    }
}

void PixelDrawingBoard::setPixel(int x, int y, uchar color)
{
    assert(x >= 0 && x < image_.width() && y >= 0 && y < image_.height());
    uchar& p = image_.scanLine(y)[x];
    if (p != color)
    {
        p = color;
        update();
    }
}

QRectF PixelDrawingBoard::pixelRect(int x, int y) const
{
    assert(x >= 0 && x < image_.width() && y >= 0 && y < image_.height());
    return QRectF(
        (qreal) x * pixelSize_.width(),
        (qreal) y * pixelSize_.height(),
        pixelSize_.width(),
        pixelSize_.height());
}

void PixelDrawingBoard::enterEvent(QEnterEvent* event)
{
    cursorPos_ = event->position();
    isEntered_ = true;
    updateBrushPoints();
    update();
}

void PixelDrawingBoard::leaveEvent(QEvent* event)
{
    isEntered_ = false;
    brushPts_.clear();
    update();
}

void PixelDrawingBoard::mouseMoveEvent(QMouseEvent* event)
{
    cursorPos_ = event->position();
    updateBrushPoints();
    if (event->buttons() & Qt::LeftButton)
        applyBrushColor();
    update();
}

void PixelDrawingBoard::mousePressEvent(QMouseEvent* event)
{
    cursorPos_ = event->position();
    updateBrushPoints();
    applyBrushColor();
    update();
}

void PixelDrawingBoard::wheelEvent(QWheelEvent* event)
{
    if (event->modifiers() & Qt::ControlModifier)
    {
        const int delta = event->angleDelta().y() > 0 ? 1 : -1;
        setBrushSize(brushSize_ + delta);
        event->accept();
    }
}

void PixelDrawingBoard::resizeEvent(QResizeEvent* event)
{
    QWidget::resizeEvent(event);
    updatePixelSize();
    if (isEntered_) updateBrushPoints();
    update();
}

void PixelDrawingBoard::paintEvent(QPaintEvent* event)
{
    QPainter painter(this);

    // 填充网格颜色
    for (int y = 0; y < image_.height(); ++y)
    {
        for (int x = 0; x < image_.width(); ++x)
        {
            const uchar v = pixel(x, y);
            painter.fillRect(pixelRect(x, y), QColor(qRgb(v, v, v)));
        }
    }

    // 绘制网格边框线
    if (gridLineWidth_ != 0.0)
    {
        painter.setPen(QPen(QBrush(gridLineColor_), gridLineWidth_));

        qreal lineY = 0.0;
        for (int y = 0; y <= image_.height(); ++y)
        {
            painter.drawLine(QPointF(0.0, lineY), QPointF((qreal) width(), lineY));
            lineY += pixelSize_.height();
        }

        qreal lineX = 0.0;
        for (int x = 0; x <= image_.width(); ++x)
        {
            painter.drawLine(QPointF(lineX, 0.0), QPointF(lineX, (qreal) height()));
            lineX += pixelSize_.width();
        }
    }

    // 绘制笔刷
    if (isEntered_)
    {
        for (const auto& pt : brushPts_)
        {
            uchar v = pixel(pt.x(), pt.y());
            v = (uchar) std::clamp((int) v + brushColor_, 0, UCHAR_MAX);
            painter.fillRect(pixelRect(pt.x(), pt.y()), QColor(qRgb(v, v, v)));
        }
    }
}

void PixelDrawingBoard::updatePixelSize()
{
    if (image_.width() == 0 || image_.height() == 0)
    {
        pixelSize_ = QSizeF();
    }
    else
    {
        pixelSize_.setWidth((qreal) width() / (qreal) image_.width());
        pixelSize_.setHeight((qreal) height() / (qreal) image_.height());
    }
}

void PixelDrawingBoard::updateBrushPoints()
{
    if (pixelSize_.width() == 0.0 || pixelSize_.height() == 0.0)
    {
        brushPts_.clear();
        update();
    }

    brushPts_.clear();
    const int px = (int) (cursorPos_.x() / pixelSize_.width());
    const int py = (int) (cursorPos_.y() / pixelSize_.height());
    const int xStart = (px - brushSize_) < 0 ? 0 : (px - brushSize_);
    const int yStart = (py - brushSize_) < 0 ? 0 : (py - brushSize_);
    const int xEnd   = (px + brushSize_) > image_.width()  ? image_.width()  : (px + brushSize_);
    const int yEnd   = (py + brushSize_) > image_.height() ? image_.height() : (py + brushSize_);
    for (int y = yStart; y < yEnd; ++y)
    {
        for (int x = xStart; x < xEnd; ++x)
        {
            const qreal dist = sqrt((qreal) ((px - x) * (px - x) + (py - y) * (py - y)));
            if (dist < (qreal) brushSize_)
                brushPts_.append(QPoint(x, y));
        }
    }
}

void PixelDrawingBoard::applyBrushColor()
{
    for (const auto& pt : brushPts_)
    {
        assert(pt.x() >= 0 && pt.x() < image_.width() && pt.y() >= 0 && pt.y() < image_.height());
        uchar* line = image_.scanLine(pt.y());
        line[pt.x()] = (uchar) std::clamp((int) line[pt.x()] + brushColor_, 0, UCHAR_MAX);
    }
}
