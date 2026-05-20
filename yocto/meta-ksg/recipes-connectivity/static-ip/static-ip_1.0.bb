SUMMARY = "Static IP configuration for Ethernet"
DESCRIPTION = "Recipe to configure a static IP address for the Ethernet interface using systemd-networkd."
LICENSE = "MIT"
LIC_FILES_CHKSUM = "file://${COMMON_LICENSE_DIR}/MIT;md5=0835ade698e0bcf8506ecda2f7b4f302"

SRC_URI = "file://10-eth0-static.network"

S = "${WORKDIR}"

do_install() {
    install -d ${D}${sysconfdir}/systemd/network
    install -m 0644 ${WORKDIR}/10-eth0-static.network ${D}${sysconfdir}/systemd/network/
}

FILES:${PN} = "${sysconfdir}/systemd/network/10-eth0-static.network"

inherit features_check
REQUIRED_DISTRO_FEATURES = "systemd"
