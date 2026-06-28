#include <qapplication.h>

#include "main_widget.h"

int main(int argc, char* argv[])
{
    QApplication a(argc, argv);
    a.setApplicationName("Digit Recognizer GUI");
    a.setApplicationDisplayName("Digit Recognizer GUI");

    MainWidget w;
    w.show();
    w.setFixedSize(w.size());

    return a.exec();
}
