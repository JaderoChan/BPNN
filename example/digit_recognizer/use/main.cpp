#include <qapplication.h>

#include "main_widget.h"

int main(int argc, char* argv[])
{
    QApplication a(argc, argv);

    MainWidget w;
    w.show();
    w.setFixedSize(w.size());

    return a.exec();
}
