#pragma once

#include <climits>

#include <qevent.h>
#include <qimage.h>
#include <qlist.h>
#include <qpainter.h>
#include <qwidget.h>

class PixelDrawingBoard : public QWidget
{
    Q_OBJECT

public:
    explicit PixelDrawingBoard(QWidget* parent = nullptr);
    explicit PixelDrawingBoard(const QImage& image, QWidget* parent = nullptr);
    explicit PixelDrawingBoard(const QSize& boardSize, uchar fillColor = UCHAR_MAX,
        QWidget* parent = nullptr);
    explicit PixelDrawingBoard(int width, int height, uchar fillColor = UCHAR_MAX,
        QWidget* parent = nullptr);

    QImage exportImage() const;
    void   importImage(const QImage& image, bool keepGridSize);
    void   fill(uchar color = UCHAR_MAX);

    QSize  boardSize() const;
    qreal  gridLineWidth() const;
    QColor gridLineColor() const;
    int    brushSize() const;
    int    brushDelta() const;
    uchar  pixel(int x, int y) const;

    void   setBoardSize(const QSize& size, uchar fillColor = UCHAR_MAX);
    void   setBoardSize(int width, int height, uchar fillColor = UCHAR_MAX);
    void   setGridLineWidth(qreal width);
    void   setGridLineColor(const QColor& color);
    void   setBrushSize(int size);
    void   setBrushDelta(int delta);
    void   setPixel(int x, int y, uchar color);

    QRectF pixelRect(int x, int y) const;

signals:
    void drawingChanged();

protected:
    void enterEvent(QEnterEvent* event) override;
    void leaveEvent(QEvent* event) override;
    void mouseMoveEvent(QMouseEvent* event) override;
    void mousePressEvent(QMouseEvent* event) override;
    void wheelEvent(QWheelEvent* event) override;
    void resizeEvent(QResizeEvent* event) override;
    void paintEvent(QPaintEvent* event) override;

private:
    void updatePixelSize();
    void updateBrushPoints();
    void applyBrushDelta();

    QImage        image_;
    int           brushSize_     = 0;
    int           brushDelta_    = 0;
    qreal         gridLineWidth_ = 0.1;
    QColor        gridLineColor_ = Qt::white;

    QSizeF        pixelSize_;
    bool          mouseInside_   = false;
    QPointF       cursorPos_;
    QList<QPoint> brushPoints_;
};
