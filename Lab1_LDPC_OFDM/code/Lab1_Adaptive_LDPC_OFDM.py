#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#
# SPDX-License-Identifier: GPL-3.0
#
# GNU Radio Python Flow Graph
# Title: Complexity-Adaptive LDPC-Coded OFDM
# Author: Balraj Minhas
# GNU Radio version: 3.10.9.2

from PyQt5 import Qt
from gnuradio import qtgui
from PyQt5 import QtCore
from PyQt5.QtCore import QObject, pyqtSlot
from gnuradio import gr
from gnuradio.filter import firdes
from gnuradio.fft import window
import sys
import signal
from PyQt5 import Qt
from argparse import ArgumentParser
from gnuradio.eng_arg import eng_float, intx
from gnuradio import eng_notation
import Lab1_Adaptive_LDPC_OFDM_epy_block_0 as epy_block_0  # embedded python block
import sip



class Lab1_Adaptive_LDPC_OFDM(gr.top_block, Qt.QWidget):

    def __init__(self):
        gr.top_block.__init__(self, "Complexity-Adaptive LDPC-Coded OFDM", catch_exceptions=True)
        Qt.QWidget.__init__(self)
        self.setWindowTitle("Complexity-Adaptive LDPC-Coded OFDM")
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

        self.settings = Qt.QSettings("GNU Radio", "Lab1_Adaptive_LDPC_OFDM")

        try:
            geometry = self.settings.value("geometry")
            if geometry:
                self.restoreGeometry(geometry)
        except BaseException as exc:
            print(f"Qt GUI: Could not restore geometry: {str(exc)}", file=sys.stderr)

        ##################################################
        # Variables
        ##################################################
        self.snr_db = snr_db = 10
        self.samp_rate = samp_rate = 64000
        self.rate_sel = rate_sel = 0
        self.max_iters = max_iters = 20
        self.jsr_db = jsr_db = -20

        ##################################################
        # Blocks
        ##################################################

        self._snr_db_range = qtgui.Range(-2, 24, 1, 10, 200)
        self._snr_db_win = qtgui.RangeWidget(self._snr_db_range, self.set_snr_db, "SNR (dB)", "counter_slider", float, QtCore.Qt.Horizontal)
        self.top_grid_layout.addWidget(self._snr_db_win, 0, 0, 1, 1)
        for r in range(0, 1):
            self.top_grid_layout.setRowStretch(r, 1)
        for c in range(0, 1):
            self.top_grid_layout.setColumnStretch(c, 1)
        # Create the options list
        self._rate_sel_options = [0, 1, 2]
        # Create the labels list
        self._rate_sel_labels = ['1/2', '2/3', '3/4']
        # Create the combo box
        self._rate_sel_tool_bar = Qt.QToolBar(self)
        self._rate_sel_tool_bar.addWidget(Qt.QLabel("LDPC Code Rate" + ": "))
        self._rate_sel_combo_box = Qt.QComboBox()
        self._rate_sel_tool_bar.addWidget(self._rate_sel_combo_box)
        for _label in self._rate_sel_labels: self._rate_sel_combo_box.addItem(_label)
        self._rate_sel_callback = lambda i: Qt.QMetaObject.invokeMethod(self._rate_sel_combo_box, "setCurrentIndex", Qt.Q_ARG("int", self._rate_sel_options.index(i)))
        self._rate_sel_callback(self.rate_sel)
        self._rate_sel_combo_box.currentIndexChanged.connect(
            lambda i: self.set_rate_sel(self._rate_sel_options[i]))
        # Create the radio buttons
        self.top_grid_layout.addWidget(self._rate_sel_tool_bar, 0, 3, 1, 1)
        for r in range(0, 1):
            self.top_grid_layout.setRowStretch(r, 1)
        for c in range(3, 4):
            self.top_grid_layout.setColumnStretch(c, 1)
        self._max_iters_range = qtgui.Range(5, 30, 1, 20, 200)
        self._max_iters_win = qtgui.RangeWidget(self._max_iters_range, self.set_max_iters, "Maximum LDPC Iterations", "counter_slider", float, QtCore.Qt.Horizontal)
        self.top_grid_layout.addWidget(self._max_iters_win, 0, 2, 1, 1)
        for r in range(0, 1):
            self.top_grid_layout.setRowStretch(r, 1)
        for c in range(2, 3):
            self.top_grid_layout.setColumnStretch(c, 1)
        self._jsr_db_range = qtgui.Range(-40, 5, 1, -20, 200)
        self._jsr_db_win = qtgui.RangeWidget(self._jsr_db_range, self.set_jsr_db, "Narrowband JSR (dB)", "counter_slider", float, QtCore.Qt.Horizontal)
        self.top_grid_layout.addWidget(self._jsr_db_win, 0, 1, 1, 1)
        for r in range(0, 1):
            self.top_grid_layout.setRowStretch(r, 1)
        for c in range(1, 2):
            self.top_grid_layout.setColumnStretch(c, 1)
        self.qtgui_waterfall_sink_x_0 = qtgui.waterfall_sink_c(
            1024, #size
            window.WIN_BLACKMAN_hARRIS, #wintype
            0, #fc
            samp_rate, #bw
            'Received OFDM waterfall', #name
            1, #number of inputs
            None # parent
        )
        self.qtgui_waterfall_sink_x_0.set_update_time(0.20)
        self.qtgui_waterfall_sink_x_0.enable_grid(True)
        self.qtgui_waterfall_sink_x_0.enable_axis_labels(True)



        labels = ['', '', '', '', '',
                  '', '', '', '', '']
        colors = [0, 0, 0, 0, 0,
                  0, 0, 0, 0, 0]
        alphas = [1.0, 1.0, 1.0, 1.0, 1.0,
                  1.0, 1.0, 1.0, 1.0, 1.0]

        for i in range(1):
            if len(labels[i]) == 0:
                self.qtgui_waterfall_sink_x_0.set_line_label(i, "Data {0}".format(i))
            else:
                self.qtgui_waterfall_sink_x_0.set_line_label(i, labels[i])
            self.qtgui_waterfall_sink_x_0.set_color_map(i, colors[i])
            self.qtgui_waterfall_sink_x_0.set_line_alpha(i, alphas[i])

        self.qtgui_waterfall_sink_x_0.set_intensity_range(-80, 20)

        self._qtgui_waterfall_sink_x_0_win = sip.wrapinstance(self.qtgui_waterfall_sink_x_0.qwidget(), Qt.QWidget)

        self.top_grid_layout.addWidget(self._qtgui_waterfall_sink_x_0_win, 3, 0, 2, 2)
        for r in range(3, 5):
            self.top_grid_layout.setRowStretch(r, 1)
        for c in range(0, 2):
            self.top_grid_layout.setColumnStretch(c, 1)
        self.qtgui_number_sink_0 = qtgui.number_sink(
            gr.sizeof_float,
            0,
            qtgui.NUM_GRAPH_NONE,
            10,
            None # parent
        )
        self.qtgui_number_sink_0.set_update_time(0.25)
        self.qtgui_number_sink_0.set_title('LDPC Receiver performance')

        labels = ['Fixed BER', 'Adaptive BER', 'Fixed FER', 'Adaptive FER', 'Fixed Avg Iterations',
            'Adaptive Avg Iterations', 'Fixed Decoder work', 'Adaptive Decoder work', 'Decoder work saved', 'Frames']
        units = ['', '', '', '', '',
            '', '', '', '', '']
        colors = [("black", "black"), ("black", "black"), ("black", "black"), ("black", "black"), ("blue", "red"),
            ("black", "black"), ("black", "black"), ("black", "black"), ("black", "black"), ("black", "black")]
        factor = [1, 1, 1, 1, 1,
            1, 1, 1, 1, 1]

        for i in range(10):
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
        self.top_grid_layout.addWidget(self._qtgui_number_sink_0_win, 3, 2, 2, 2)
        for r in range(3, 5):
            self.top_grid_layout.setRowStretch(r, 1)
        for c in range(2, 4):
            self.top_grid_layout.setColumnStretch(c, 1)
        self.qtgui_freq_sink_x_0 = qtgui.freq_sink_c(
            1024, #size
            window.WIN_BLACKMAN_hARRIS, #wintype
            0, #fc
            samp_rate, #bw
            'Received OFDM Spectrum with Narrowband Interference', #name
            1,
            None # parent
        )
        self.qtgui_freq_sink_x_0.set_update_time(0.20)
        self.qtgui_freq_sink_x_0.set_y_axis((-80), 20)
        self.qtgui_freq_sink_x_0.set_y_label('Relative Gain', 'dB')
        self.qtgui_freq_sink_x_0.set_trigger_mode(qtgui.TRIG_MODE_FREE, 0.0, 0, "")
        self.qtgui_freq_sink_x_0.enable_autoscale(False)
        self.qtgui_freq_sink_x_0.enable_grid(True)
        self.qtgui_freq_sink_x_0.set_fft_average(0.2)
        self.qtgui_freq_sink_x_0.enable_axis_labels(True)
        self.qtgui_freq_sink_x_0.enable_control_panel(False)
        self.qtgui_freq_sink_x_0.set_fft_window_normalized(False)



        labels = ['', '', '', '', '',
            '', '', '', '', '']
        widths = [1, 1, 1, 1, 1,
            1, 1, 1, 1, 1]
        colors = ["blue", "red", "green", "black", "cyan",
            "magenta", "yellow", "dark red", "dark green", "dark blue"]
        alphas = [1.0, 1.0, 1.0, 1.0, 1.0,
            1.0, 1.0, 1.0, 1.0, 1.0]

        for i in range(1):
            if len(labels[i]) == 0:
                self.qtgui_freq_sink_x_0.set_line_label(i, "Data {0}".format(i))
            else:
                self.qtgui_freq_sink_x_0.set_line_label(i, labels[i])
            self.qtgui_freq_sink_x_0.set_line_width(i, widths[i])
            self.qtgui_freq_sink_x_0.set_line_color(i, colors[i])
            self.qtgui_freq_sink_x_0.set_line_alpha(i, alphas[i])

        self._qtgui_freq_sink_x_0_win = sip.wrapinstance(self.qtgui_freq_sink_x_0.qwidget(), Qt.QWidget)
        self.top_grid_layout.addWidget(self._qtgui_freq_sink_x_0_win, 1, 2, 2, 2)
        for r in range(1, 3):
            self.top_grid_layout.setRowStretch(r, 1)
        for c in range(2, 4):
            self.top_grid_layout.setColumnStretch(c, 1)
        self.qtgui_const_sink_x_0 = qtgui.const_sink_c(
            1024, #size
            'TX vs Equalized RX Constellation', #name
            2, #number of inputs
            None # parent
        )
        self.qtgui_const_sink_x_0.set_update_time(0.20)
        self.qtgui_const_sink_x_0.set_y_axis((-2.5), 2)
        self.qtgui_const_sink_x_0.set_x_axis((-2.5), 2.5)
        self.qtgui_const_sink_x_0.set_trigger_mode(qtgui.TRIG_MODE_FREE, qtgui.TRIG_SLOPE_POS, 0.0, 0, "")
        self.qtgui_const_sink_x_0.enable_autoscale(False)
        self.qtgui_const_sink_x_0.enable_grid(True)
        self.qtgui_const_sink_x_0.enable_axis_labels(True)


        labels = ['Ideal TX QPSK', 'Equalized RX', '', '', '',
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

        for i in range(2):
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
        self.epy_block_0 = epy_block_0.blk(snr_db=snr_db, jsr_db=jsr_db, max_iters=max_iters, rate_sel=rate_sel, seed=2026, window_frames=1000, report_mode=True, frames_per_point=1000, extended_report=True, csi_mode="perfect", code_family="random_eira", quant_bits=0, run_label="gnuradio")


        ##################################################
        # Connections
        ##################################################
        self.connect((self.epy_block_0, 0), (self.qtgui_const_sink_x_0, 0))
        self.connect((self.epy_block_0, 1), (self.qtgui_const_sink_x_0, 1))
        self.connect((self.epy_block_0, 2), (self.qtgui_freq_sink_x_0, 0))
        self.connect((self.epy_block_0, 8), (self.qtgui_number_sink_0, 5))
        self.connect((self.epy_block_0, 6), (self.qtgui_number_sink_0, 3))
        self.connect((self.epy_block_0, 5), (self.qtgui_number_sink_0, 2))
        self.connect((self.epy_block_0, 10), (self.qtgui_number_sink_0, 7))
        self.connect((self.epy_block_0, 7), (self.qtgui_number_sink_0, 4))
        self.connect((self.epy_block_0, 11), (self.qtgui_number_sink_0, 8))
        self.connect((self.epy_block_0, 3), (self.qtgui_number_sink_0, 0))
        self.connect((self.epy_block_0, 9), (self.qtgui_number_sink_0, 6))
        self.connect((self.epy_block_0, 4), (self.qtgui_number_sink_0, 1))
        self.connect((self.epy_block_0, 12), (self.qtgui_number_sink_0, 9))
        self.connect((self.epy_block_0, 2), (self.qtgui_waterfall_sink_x_0, 0))


    def closeEvent(self, event):
        self.settings = Qt.QSettings("GNU Radio", "Lab1_Adaptive_LDPC_OFDM")
        self.settings.setValue("geometry", self.saveGeometry())
        self.stop()
        self.wait()

        event.accept()

    def get_snr_db(self):
        return self.snr_db

    def set_snr_db(self, snr_db):
        self.snr_db = snr_db
        self.epy_block_0.snr_db = self.snr_db

    def get_samp_rate(self):
        return self.samp_rate

    def set_samp_rate(self, samp_rate):
        self.samp_rate = samp_rate
        self.qtgui_freq_sink_x_0.set_frequency_range(0, self.samp_rate)
        self.qtgui_waterfall_sink_x_0.set_frequency_range(0, self.samp_rate)

    def get_rate_sel(self):
        return self.rate_sel

    def set_rate_sel(self, rate_sel):
        self.rate_sel = rate_sel
        self._rate_sel_callback(self.rate_sel)
        self.epy_block_0.rate_sel = self.rate_sel

    def get_max_iters(self):
        return self.max_iters

    def set_max_iters(self, max_iters):
        self.max_iters = max_iters
        self.epy_block_0.max_iters = self.max_iters

    def get_jsr_db(self):
        return self.jsr_db

    def set_jsr_db(self, jsr_db):
        self.jsr_db = jsr_db
        self.epy_block_0.jsr_db = self.jsr_db




def main(top_block_cls=Lab1_Adaptive_LDPC_OFDM, options=None):

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
