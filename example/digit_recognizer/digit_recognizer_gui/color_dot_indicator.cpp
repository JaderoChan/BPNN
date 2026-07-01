#include "color_dot_indicator.h"

#include <qpainter.h>

ColorDotIndicator::ColorDotIndicator(const QColor& color, QWidget* parent)
    : QWidget(parent), color_(color)
{}

QColor ColorDotIndicator::color()
{
    return color_;
}

void ColorDotIndicator::setColor(const QColor& color)
{
    color_ = color;
    update();
    emit colorChanged(color);
}

QSize ColorDotIndicator::sizeHint() const
{
    return QSize(16, 16);
}

void ColorDotIndicator::paintEvent(QPaintEvent* event)
{
    QPainter painter(this);
    painter.setBrush(color_);
    painter.setPen(Qt::transparent);
    painter.drawEllipse(rect());
}
