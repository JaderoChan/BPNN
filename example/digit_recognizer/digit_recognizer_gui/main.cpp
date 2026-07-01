#include <qapplication.h>

#include "main_widget.h"

#define APP_NAME "Digit Recognizer GUI"

int main(int argc, char* argv[])
{
    QApplication a(argc, argv);
    a.setApplicationName(APP_NAME);
    a.setApplicationDisplayName(APP_NAME);

    MainWidget w;
    w.show();
    w.setFixedSize(w.size());

    return a.exec();
}
