FILESEXTRAPATHS:prepend := "${THISDIR}/files:"

SRC_URI:remove = "git://git.kernel.org/pub/scm/linux/kernel/git/next/linux-next.git;protocol=https;nobranch=1"
SRC_URI:prepend = "git://git.kernel.org/pub/scm/linux/kernel/git/next/linux-next-history.git;protocol=https;nobranch=1 "

SRC_URI:append = " \
    file://camera.cfg \
    file://0001-dts-add-imx219-camera-sensor.patch \
"
