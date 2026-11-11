#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#
# SPDX-License-Identifier: GPL-3.0
#
# GNU Radio Python Flow Graph
# Title: 2x2 MIMO-OFDM Detection under Channel Correlation and Estimation Error
# Author: Balraj Minhas
# Description: 2x2 spatial multiplexing, correlated Rayleigh fading, pilot LS CSI, ZF/MMSE/MMSE-SIC
# GNU Radio version: 3.10.9.2

from PyQt5 import Qt
from gnuradio import qtgui
from PyQt5 import QtCore
from gnuradio import gr
from gnuradio.filter import firdes
from gnuradio.fft import window
import sys
import signal
from PyQt5 import Qt
from argparse import ArgumentParser
from gnuradio.eng_arg import eng_float, intx
from gnuradio import eng_notation
import Lab2_MIMO_OFDM_epy_block_0 as epy_block_0  # embedded python block
import sip



class Lab2_MIMO_OFDM(gr.top_block, Qt.QWidget):

    def __init__(self):
        gr.top_block.__init__(self, "2x2 MIMO-OFDM Detection under Channel Correlation and Estimation Error", catch_exceptions=True)
        Qt.QWidget.__init__(self)
        self.setWindowTitle("2x2 MIMO-OFDM Detection under Channel Correlation and Estimation Error")
        qtgui.util.check_set_qss()
        try:
            self.setWindowIcon(Qt.QIcon.fromTheme('gnuradio-grc'))
        except BaseException as exc:
            print(f"Qt GUI: Could not set Icon: {str(exc)}", file=sys.stderr)
        self.top_scroll_layout = Qt.QVBoxLayout()
        self.setLayout(self.top_scroll_layout)
        self.top_scroll = Qt.QScrollArea()
        self.top_scroll.setFrameStyle(Qt.QFrame.NoFrame)
        self.top_scroll_layout.addWidget(self.top_scroll)
        self.top_scroll.setWidgetResizable(True)
        self.top_widget = Qt.QWidget()
        self.top_scroll.setWidget(self.top_widget)
        self.top_layout = Qt.QVBoxLayout(self.top_widget)
        self.top_grid_layout = Qt.QGridLayout()
        self.top_layout.addLayout(self.top_grid_layout)

        self.settings = Qt.QSettings("GNU Radio", "Lab2_MIMO_OFDM")

        try:
            geometry = self.settings.value("geometry")
            if geometry:
                self.restoreGeometry(geometry)
        except BaseException as exc:
            print(f"Qt GUI: Could not restore geometry: {str(exc)}", file=sys.stderr)

        ##################################################
        # Variables
        ##################################################
        self.snr_db = snr_db = 12
        self.rho = rho = 0.6
        self.pilot_snr_db = pilot_snr_db = 20

        ##################################################
        # Blocks
        ##################################################

        self._snr_db_range = qtgui.Range(0, 24, 1, 12, 200)
        self._snr_db_win = qtgui.RangeWidget(self._snr_db_range, self.set_snr_db, "Data Es/N0 (dB)", "counter_slider", float, QtCore.Qt.Horizontal)
        self.top_grid_layout.addWidget(self._snr_db_win, 0, 0, 1, 1)
        for r in range(0, 1):
            self.top_grid_layout.setRowStretch(r, 1)
        for c in range(0, 1):
            self.top_grid_layout.setColumnStretch(c, 1)
        self._rho_range = qtgui.Range(0, 0.95, 0.05, 0.6, 200)
        self._rho_win = qtgui.RangeWidget(self._rho_range, self.set_rho, "Spatial correlation rho", "counter_slider", float, QtCore.Qt.Horizontal)
        self.top_grid_layout.addWidget(self._rho_win, 0, 1, 1, 1)
        for r in range(0, 1):
            self.top_grid_layout.setRowStretch(r, 1)
        for c in range(1, 2):
            self.top_grid_layout.setColumnStretch(c, 1)
        self._pilot_snr_db_range = qtgui.Range(0, 30, 2, 20, 200)
        self._pilot_snr_db_win = qtgui.RangeWidget(self._pilot_snr_db_range, self.set_pilot_snr_db, "Pilot Es/N0 (dB)", "counter_slider", float, QtCore.Qt.Horizontal)
        self.top_grid_layout.addWidget(self._pilot_snr_db_win, 0, 2, 1, 1)
        for r in range(0, 1):
            self.top_grid_layout.setRowStretch(r, 1)
        for c in range(2, 3):
            self.top_grid_layout.setColumnStretch(c, 1)
        self.qtgui_number_sink_0 = qtgui.number_sink(
            gr.sizeof_float,
            0,
            qtgui.NUM_GRAPH_NONE,
            5,
            None # parent
        )
        self.qtgui_number_sink_0.set_update_time(0.25)
        self.qtgui_number_sink_0.set_title('Instantaneous frame metrics (GUI only)')

        labels = ['ZF frame BER', 'MMSE frame BER', 'MMSE-SIC frame BER', 'Median channel condition', 'LS channel NMSE',
            'Adaptive Avg Iterations', 'Fixed Decoder work', 'Adaptive Decoder work', 'Decoder work saved', 'Frames']
        units = ['', '', '', '', '',
            '', '', '', '', '']
        colors = [("black", "black"), ("black", "black"), ("black", "black"), ("black", "black"), ("blue", "red"),
            ("black", "black"), ("black", "black"), ("black", "black"), ("black", "black"), ("black", "black")]
        factor = [1, 1, 1, 1, 1,
            1, 1, 1, 1, 1]

        for i in range(5):
            self.qtgui_number_sink_0.set_min(i, -1)
            self.qtgui_number_sink_0.set_max(i, 1)
            self.qtgui_number_sink_0.set_color(i, colors[i][0], colors[i][1])
            if len(labels[i]) == 0:
                self.qtgui_number_sink_0.set_label(i, "Data {0}".format(i))
            else:
                self.qtgui_number_sink_0.set_label(i, labels[i])
            self.qtgui_number_sink_0.set_unit(i, units[i])
            self.qtgui_number_sink_0.set_factor(i, factor[i])

        self.qtgui_number_sink_0.enable_autoscale(True)
        self._qtgui_number_sink_0_win = sip.wrapinstance(self.qtgui_number_sink_0.qwidget(), Qt.QWidget)
        self.top_grid_layout.addWidget(self._qtgui_number_sink_0_win, 3, 0, 2, 2)
        for r in range(3, 5):
            self.top_grid_layout.setRowStretch(r, 1)
        for c in range(0, 2):
            self.top_grid_layout.setColumnStretch(c, 1)
        self.qtgui_const_sink_x_0 = qtgui.const_sink_c(
            1024, #size
            'QPSK: transmitted / ZF / MMSE / MMSE-SIC', #name
            4, #number of inputs
            None # parent
        )
        self.qtgui_const_sink_x_0.set_update_time(0.20)
        self.qtgui_const_sink_x_0.set_y_axis((-2.5), 2)
        self.qtgui_const_sink_x_0.set_x_axis((-2.5), 2.5)
        self.qtgui_const_sink_x_0.set_trigger_mode(qtgui.TRIG_MODE_FREE, qtgui.TRIG_SLOPE_POS, 0.0, 0, "")
        self.qtgui_const_sink_x_0.enable_autoscale(False)
        self.qtgui_const_sink_x_0.enable_grid(True)
        self.qtgui_const_sink_x_0.enable_axis_labels(True)


        labels = ['Transmitted QPSK', 'ZF', 'MMSE', 'MMSE-SIC', '',
            '', '', '', '', '']
        widths = [1, 1, 1, 1, 1,
            1, 1, 1, 1, 1]
        colors = ["blue", "red", "green", "black", "cyan",
            "magenta", "yellow", "dark red", "dark green", "dark blue"]
        styles = [0, 0, 0, 0, 0,
            0, 0, 0, 0, 0]
        markers = [0, 0, 0, 0, 0,
            0, 0, 0, 0, 0]
        alphas = [1.0, 1.0, 1.0, 1.0, 1.0,
            1.0, 1.0, 1.0, 1.0, 1.0]

        for i in range(4):
            if len(labels[i]) == 0:
                self.qtgui_const_sink_x_0.set_line_label(i, "Data {0}".format(i))
            else:
                self.qtgui_const_sink_x_0.set_line_label(i, labels[i])
            self.qtgui_const_sink_x_0.set_line_width(i, widths[i])
            self.qtgui_const_sink_x_0.set_line_color(i, colors[i])
            self.qtgui_const_sink_x_0.set_line_style(i, styles[i])
            self.qtgui_const_sink_x_0.set_line_marker(i, markers[i])
            self.qtgui_const_sink_x_0.set_line_alpha(i, alphas[i])

        self._qtgui_const_sink_x_0_win = sip.wrapinstance(self.qtgui_const_sink_x_0.qwidget(), Qt.QWidget)
        self.top_grid_layout.addWidget(self._qtgui_const_sink_x_0_win, 1, 0, 2, 2)
        for r in range(1, 3):
            self.top_grid_layout.setRowStretch(r, 1)
        for c in range(0, 2):
            self.top_grid_layout.setColumnStretch(c, 1)
        self.epy_block_0 = epy_block_0.blk(snr_db=snr_db, rho=rho, pilot_snr_db=pilot_snr_db, seed=2026)


        ##################################################
        # Connections
        ##################################################
        self.connect((self.epy_block_0, 3), (self.qtgui_const_sink_x_0, 3))
        self.connect((self.epy_block_0, 0), (self.qtgui_const_sink_x_0, 0))
        self.connect((self.epy_block_0, 2), (self.qtgui_const_sink_x_0, 2))
        self.connect((self.epy_block_0, 1), (self.qtgui_const_sink_x_0, 1))
        self.connect((self.epy_block_0, 5), (self.qtgui_number_sink_0, 1))
        self.connect((self.epy_block_0, 4), (self.qtgui_number_sink_0, 0))
        self.connect((self.epy_block_0, 7), (self.qtgui_number_sink_0, 3))
        self.connect((self.epy_block_0, 8), (self.qtgui_number_sink_0, 4))
        self.connect((self.epy_block_0, 6), (self.qtgui_number_sink_0, 2))


    def closeEvent(self, event):
        self.settings = Qt.QSettings("GNU Radio", "Lab2_MIMO_OFDM")
        self.settings.setValue("geometry", self.saveGeometry())
        self.stop()
        self.wait()

        event.accept()

    def get_snr_db(self):
        return self.snr_db

    def set_snr_db(self, snr_db):
        self.snr_db = snr_db
        self.epy_block_0.snr_db = self.snr_db

    def get_rho(self):
        return self.rho

    def set_rho(self, rho):
        self.rho = rho
        self.epy_block_0.rho = self.rho

    def get_pilot_snr_db(self):
        return self.pilot_snr_db

    def set_pilot_snr_db(self, pilot_snr_db):
        self.pilot_snr_db = pilot_snr_db
        self.epy_block_0.pilot_snr_db = self.pilot_snr_db




def main(top_block_cls=Lab2_MIMO_OFDM, options=None):

    qapp = Qt.QApplication(sys.argv)

    tb = top_block_cls()

    tb.start()

    tb.show()

    def sig_handler(sig=None, frame=None):
        tb.stop()
        tb.wait()

        Qt.QApplication.quit()

    signal.signal(signal.SIGINT, sig_handler)
    signal.signal(signal.SIGTERM, sig_handler)

    timer = Qt.QTimer()
    timer.start(500)
    timer.timeout.connect(lambda: None)

    qapp.exec_()

if __name__ == '__main__':
    main()
