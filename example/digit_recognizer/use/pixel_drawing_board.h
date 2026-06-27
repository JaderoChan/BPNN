#include <climits>

#include <qevent.h>
#include <qlist.h>
#include <qpainter.h>
#include <qimage.h>
#include <qwidget.h>

class PixelDrawingBoard : public QWidget
{
public:
    explicit PixelDrawingBoard(QWidget* parent = nullptr);
    explicit PixelDrawingBoard(const QImage& image, QWidget* parent = nullptr);
    explicit PixelDrawingBoard(
        const QSize& boardSize, uchar fillColor = UCHAR_MAX,
        QWidget* parent = nullptr);

    QImage exportImage() const;
    void   importImage(const QImage& image);
    void   fill(uchar color = UCHAR_MAX);

    QSize  boardSize() const;
    qreal  gridLineWidth() const;
    QColor gridLineColor() const;
    int    brushSize() const;
    int    brushColor() const;
    uchar  pixel(int x, int y) const;

    void   setBoardSize(const QSize& size, uchar fillColor = UCHAR_MAX);
    void   setGridLineWidth(qreal value);
    void   setGridLineColor(const QColor& color);
    void   setBrushSize(int value);
    void   setBrushColor(int value);
    void   setPixel(int x, int y, uchar color);

    QRectF pixelRect(int x, int y) const;

protected:
    void enterEvent(QEnterEvent* event) override;
    void leaveEvent(QEvent* event) override;
    void mouseMoveEvent(QMouseEvent* event) override;
    void mousePressEvent(QMouseEvent* event) override;
    void mouseReleaseEvent(QMouseEvent* event) override;
    void wheelEvent(QWheelEvent* event) override;
    void resizeEvent(QResizeEvent* event) override;
    void paintEvent(QPaintEvent* event) override;

private:
    void updatePixelSize();
    void updateBrushPoints();
    void applyBrushColor();

    QImage image_;
    int    brushSize_      = 5;
    int    brushColor_     = -128;
    qreal  gridLineWidth_  = 0.1;
    QColor gridLineColor_  = Qt::white;

    QSizeF pixelSize_;
    bool   isEntered_     = false;
    bool   isPressed_     = false;
    QPointF       cursorPos_;
    QList<QPoint> brushPts_;
};
