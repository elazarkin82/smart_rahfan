inherit extrausers

EXTRA_USERS_PARAMS = "\
    useradd -p '\$6\$BKzOq9NLLL7fWy/1\$u9MPE74kJBnL7FosFeYhulri797nW6bsB.77T/EAb1iOv0s8GHP0fINYfOCT8UpvkiLuArpVkJGjiOFe2L3l0/' -s /bin/sh -d /home/ksg -m ksg; \
    usermod -p '\$6\$tUt2X52o.aNjxxTu\$ZzXiDfbFJbEII.PPi5ZzZP0sUCIoK020koNej6burNH2NSPnXEOXhgrsd8Yf.oOD/FcLAJDSFDXDla7s0ed0w1' root; \
    userdel radxa; \
"

IMAGE_INSTALL:append = " rknpu2 rockchip-librga libgomp"
