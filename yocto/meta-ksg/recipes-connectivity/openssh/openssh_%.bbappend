do_install:append() {
    # Disable root login over SSH for security
    sed -i 's/#PermitRootLogin.*/PermitRootLogin no/' ${D}${sysconfdir}/ssh/sshd_config
    sed -i 's/PermitRootLogin.*/PermitRootLogin no/' ${D}${sysconfdir}/ssh/sshd_config
    if [ -f ${D}${sysconfdir}/ssh/sshd_config_readonly ]; then
        sed -i 's/#PermitRootLogin.*/PermitRootLogin no/' ${D}${sysconfdir}/ssh/sshd_config_readonly
        sed -i 's/PermitRootLogin.*/PermitRootLogin no/' ${D}${sysconfdir}/ssh/sshd_config_readonly
    fi
}
