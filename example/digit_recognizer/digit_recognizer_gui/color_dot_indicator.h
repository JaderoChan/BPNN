#pragma once

#include <qcolor.h>
#include <qwidget.h>
#include <qevent.h>

class ColorDotIndicator : public QWidget
{
    Q_OBJECT

public:
    explicit ColorDotIndicator(const QColor& color = Qt::red, QWidget* parent = nullptr);

    QColor color();
    void   setColor(const QColor& color);

    QSize  sizeHint() const override;

signals:
    void colorChanged(QColor color);

protected:
    void paintEvent(QPaintEvent* event) override;

private:
    QColor color_;
};
