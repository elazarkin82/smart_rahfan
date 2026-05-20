DESCRIPTION = "Radxa Rockchip BSP Linux Kernel 5.10"
SECTION = "kernel"
LICENSE = "GPL-2.0-only"
LIC_FILES_CHKSUM = "file://COPYING;md5=6bc538ed5bd9a7fc9398086aedcd7e46"

inherit kernel

COMPATIBLE_MACHINE = "radxa-zero-3e"

DEPENDS += "openssl-native elfutils-native bison-native flex-native"

SRC_URI = "git://github.com/radxa/kernel.git;protocol=https;branch=linux-5.10-gen-rkr4.1"
SRCREV = "${AUTOREV}"

S = "${WORKDIR}/git"

KBUILD_DEFCONFIG = "rockchip_linux_defconfig"

KCFLAGS += "-Wno-error"
export KCFLAGS

do_configure:prepend() {
    # Generate the .config file using the in-tree defconfig
    oe_runmake -C ${S} O=${B} ${KBUILD_DEFCONFIG}

    # Disable Rockchip WLAN drivers that fail to compile under modern GCC and out-of-tree builds
    ${S}/scripts/config --file ${B}/.config --disable CONFIG_WL_ROCKCHIP
    ${S}/scripts/config --file ${B}/.config --disable CONFIG_RTL8852BE
}
