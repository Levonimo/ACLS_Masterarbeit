"""
Personalized PyQt stylesheets

Adapted from 
  - Nicola Imstepf's Master Thesis 
  - David Patsch's AlphaDock
  - Colin Duquesnoy's QDarkStyleSheet (https://github.com/ColinDuquesnoy/QDarkStyleSheet)

styles.py
"""

Levin = """

QCheckBox {
  background-color: #19232D;
  color: #E0E1E3;
  spacing: 4px;
  outline: none;
  padding-top: 4px;
  padding-bottom: 4px;
}
QCheckBox:focus {
  border: none;
}
QCheckBox QWidget:disabled {
  background-color: #19232D;
  color: #9DA9B5;
}
QCheckBox::indicator {
  margin-left: 2px;
  height: 14px;
  width: 14px;
}
QCheckBox::indicator:unchecked {
  image: url("component/images/checkbox_unchecked.png");
}
QCheckBox::indicator:unchecked:hover, QCheckBox::indicator:unchecked:focus, QCheckBox::indicator:unchecked:pressed {
  border: none;
  image: url("component/images/checkbox_unchecked_focus.png");
}
QCheckBox::indicator:unchecked:disabled {
  image: url("component/images/checkbox_unchecked_disabled.png");
}
QCheckBox::indicator:checked {
  image: url("component/images/checkbox_checked.png");
}
QCheckBox::indicator:checked:hover, QCheckBox::indicator:checked:focus, QCheckBox::indicator:checked:pressed {
  border: none;
  image: url("component/images/checkbox_checked_focus.png");
}
QCheckBox::indicator:checked:disabled {
  image: url("component/images/checkbox_checked_disabled.png");
}
QCheckBox::indicator:indeterminate {
  image: url("component/images/checkbox_indeterminate.png");
}
QCheckBox::indicator:indeterminate:disabled {
  image: url("component/images/checkbox_indeterminate_disabled.png");
}
QCheckBox::indicator:indeterminate:focus, QCheckBox::indicator:indeterminate:hover, QCheckBox::indicator:indeterminate:pressed {
  image: url("component/images/checkbox_indeterminate_focus.png");
}

QTextEdit {
  border: 1px solid white;
  color: white;
}

QTextEdit:hover {
  border: 1px solid white;
  color: white;
}

QComboBox {
  background-color: #19232D;
  border: 1px solid #32414B;
  border-radius: 4px;
  padding: 2px;
  padding-left: 5px;
  color: #E0E1E3;
  selection-color: #19232D;
  selection-background-color: #E0E1E3;
}
QComboBox QAbstractItemView {
  background-color: #19232D;
  color: #E0E1E3;
  selection-color: #19232D;
  selection-background-color: #E0E1E3;
}
QComboBox::drop-down {
  subcontrol-origin: padding;
  subcontrol-position: top right;
  width: 20px;
  border-left-width: 1px;
  border-left-color: #32414B;
  border-left-style: solid;
  border-top-right-radius: 3px;
  border-bottom-right-radius: 3px;
}
QComboBox::down-arrow {
  image: url("component/images/arrow_down.png");
}
QComboBox::down-arrow:on {
  top: 1px;
  left: 1px;
}
QComboBox QAbstractItemView {
  border: 1px solid #32414B;
  border-radius: 2px;
  background-color: #19232D;
}
QComboBox QAbstractItemView::item {
  height: 20px;
  color: #E0E1E3;
  padding-left: 5px;
  padding-top: 2px;
  padding-bottom: 2px;
}
QComboBox QAbstractItemView::item:selected {
  background-color: #32414B;
}
QComboBox QAbstractItemView::item:hover {
  background-color: #32414B;
}



QGroupBox {
	color: rgb(255, 255, 255);
	border: 2px solid rgb(40, 44, 52);
	border-top-color: rgb(255, 255, 255);
  margin-top: 10px;
  padding-top: 10px;
}
QGroupBox:disabled {
  border-top-color: rgb(69, 83, 100);
}
QGroupBox::title {
    subcontrol-origin:  margin;
	  subcontrol-position: top left;
}
QLabel {
    color: rgb(255, 255, 255);
    background-color: rgb(40, 44, 52);
}


QLineEdit {
	background-color: rgb(33, 37, 43);
	border-radius: 5px;
	border: 2px solid rgb(33, 37, 43);
	padding-left: 10px;
	selection-color: rgb(255, 255, 255);
	selection-background-color:rgb(0, 172, 255);
	color: rgb(255, 255, 255);
	padding:2px;
	padding-left: 10px;
}
QLineEdit:hover {
	border: 2px solid rgb(64, 71, 88);
	padding-left: 10px;
}
QLineEdit:focus {
	border: 2px solid rgb(91, 101, 124);
	padding-left: 10px;
}


QMenu {
    background-color: rgb(37, 41, 48);
    color: rgb(255, 255, 255);
    border: 0px;
}
QMenu::item {
    /* sets background of menu item. set this to something non-transparent
        if you want menu color and menu item color to be different */
    background-color: transparent;
    padding: 5px 40px;
    border - 0px;
}
QMenu::item:selected { /* when user selects item using mouse or keyboard */
	background-color: rgb(57, 65, 80);
	border: 2px solid rgb(61, 70, 86);
}
QMenu::item:hover { /* when user selects item using mouse or keyboard */
	background-color: rgb(57, 65, 80);
	border: 2px solid rgb(61, 70, 86);
}
QMenu::indicator {
    image: url(component/images/transparent.png);
    width: 0px;
    height: 0px;
}

QMenu::item:checked { /* when user selects item using mouse or keyboard */
  image: url(component/images/transparent.png);
  background-color: rgb(57, 65, 80);
	border: 2px solid rgb(61, 70, 86);
  background-image: url(component/images/cil-check-alt.png);
  background-repeat: no-repeat;
  background-position: center left;   
}
QMenu::icon:checked { /* when user selects item using mouse or keyboard */
	background-color: rgb(57, 65, 80);
	border: 3px solid rgb(61, 70, 86);
}


QPushButton {
	border: 1px solid rgb(57, 65, 80);
	border-radius: 5px;	
	background-color: rgb(52, 59, 72);
	color: rgb(255, 255, 255);
  padding: 2px;
}
QPushButton:hover {
	background-color: rgb(57, 65, 80);
	border: 1px solid rgb(61, 70, 86);
}
QPushButton:pressed {	
	background-color: rgb(35, 40, 49);
	border: 1px solid rgb(43, 50, 61);
}


QRadioButton {
  color: rgb(255, 255, 255);
}


QSpinBox {
  background-color: rgb(27, 29, 35);
	border-radius: 5px;
	border: 2px solid rgb(33, 37, 43);
	padding: 2px;
	padding-left: 10px;
	color: rgb(255, 255, 255);
	font: 75 10pt "Verdana";
}
QSpinBox:hover { border: 2px solid rgb(64, 71, 88);}
QSpinBox:focus { border: 2px solid rgb(91, 101, 124);}
QSpinBox::up-button {border-top-right-radius: 3px;}
QSpinBox::down-button {border-bottom-right-radius: 3px;}
QSpinBox::up-arrow { image: url(component/images/cil-arrow-up.png); width: 10px; height: 10px;}
QSpinBox::down-arrow { image: url(component/images/cil-arrow-bottom.png); width: 10px; height: 10px;}
QSpinBox::up-arrow:disabled { image: url(component/images/transparent.png); width: 10px; height: 10px;}
QSpinBox::down-arrow:disabled { image: url(component/images/transparent.png); width: 10px; height: 10px;}


QWidget {
background-color: rgb(52, 59, 72);
}
QWidget:disabled {
  color: rgb(69, 83, 100);
}
"""

MenuButton = """
QPushButton { background-color: rgba(255, 255, 255, 0); border: none;  border-radius: 5px;}
QPushButton:hover { background-color: rgb(44, 49, 57); border-style: solid; border-radius: 8px;}
QPushButton:pressed { background-color: rgb(23, 26, 30); border-style: solid; border-radius: 8px;}
"""

SettingsButton = MenuButton + """
QPushButton::menu-indicator{ image: none;}
QPushButton { qproperty-icon: url(component/images/cil-settings.png); qproperty-iconSize: 40px 40px;}
"""

MapButton = MenuButton + """
QPushButton { qproperty-icon: url(component/images/cil-map.png); qproperty-iconSize: 40px 40px;}
"""

MinButton = MenuButton + """
QPushButton { qproperty-icon: url(component/images/icon_minimize.png); qproperty-iconSize: 40px 40px;}
"""

CloseButton = MenuButton + """
QPushButton { qproperty-icon: url(component/images/icon_close.png); qproperty-iconSize: 40px 40px;}
"""

analyticlogo = MenuButton + """
QPushButton:hover { background-color: rgba(255, 255, 255, 0); border: none;  border-radius: 5px;}
QPushButton { qproperty-icon: url(component/images/Logo_ICBT_Analytik_round.png); qproperty-iconSize: 40px 40px; }
"""

DirectoryButton = """
qproperty-icon: url(component/images/cil-folder-open.png);
""" 

HoverGreenButton = """
QPushButton:hover {
	background-color:   rgb(62, 172, 111);
	border: 2px solid rgb(61, 70, 86);
	color: rgb(0, 0, 0);
}
"""

HoverPurpleButton = """
QPushButton:hover {
	background-color:   rgb(144, 87, 153);
	border: 2px solid rgb(61, 70, 86);
	color: rgb(0, 0, 0);
}
"""

LogicalButton = """
QPushButton {
	background-color: rgb(0, 119, 196);
	border: 2px solid rgb(43, 50, 61);
  color: rgb(0, 0, 0)
}
QPushButton:disabled {
  border: 2px solid rgb(52, 59, 72);
  background-color: rgb(40, 44, 52);
  color: rgb(69, 83, 100);
}
"""
