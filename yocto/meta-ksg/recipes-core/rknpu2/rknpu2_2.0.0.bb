SUMMARY = "Rockchip RK3566 NPU Runtime library"
DESCRIPTION = "Rockchip RKNN user-space runtime library (librknnrt.so) for hardware NPU acceleration."
LICENSE = "Apache-2.0"
LIC_FILES_CHKSUM = "file://${COMMON_LICENSE_DIR}/Apache-2.0;md5=89aea4e17d99a7cacdbeed46a0096b10"

SRC_URI = "file://librknnrt.so"

S = "${WORKDIR}"

# Prebuilt library, nothing to configure or compile
do_configure[noexec] = "1"
do_compile[noexec] = "1"

do_install() {
    install -d ${D}${libdir}
    install -m 0755 ${S}/librknnrt.so ${D}${libdir}/librknnrt.so
}

# Prevent packaging of .so files in the -dev package, force them into the main package
FILES_SOLIBSDEV = ""
FILES:${PN} = "${libdir}/librknnrt.so"

# Skip standard QA checks for prebuilt binaries (GNU_HASH, stripped, dev-so, dev-elf)
INSANE_SKIP:${PN} = "already-stripped ldflags dev-so dev-elf"
INSANE_SKIP:${PN}-dev = "ldflags dev-elf"

PROVIDES = "rknpu2"

