#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#
# SPDX-License-Identifier: GPL-3.0
#
# GNU Radio Python Flow Graph
# Title: Robust Chirp Acquisition and Packet Recovery under Interference
# Author: Balraj Minhas
# Description: Synthetic complex-baseband packet receiver: acquisition, symbol demodulation, CRC verification, and mitigation
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
import Lab4_Chirp_Receiver_epy_block_0 as epy_block_0  # embedded python block
import sip



class Lab4_Chirp_Receiver(gr.top_block, Qt.QWidget):

    def __init__(self):
        gr.top_block.__init__(self, "Robust Chirp Acquisition and Packet Recovery under Interference", catch_exceptions=True)
        Qt.QWidget.__init__(self)
        self.setWindowTitle("Robust Chirp Acquisition and Packet Recovery under Interference")
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

        self.settings = Qt.QSettings("GNU Radio", "Lab4_Chirp_Receiver")

        try:
            geometry = self.settings.value("geometry")
            if geometry:
                self.restoreGeometry(geometry)
        except BaseException as exc:
            print(f"Qt GUI: Could not restore geometry: {str(exc)}", file=sys.stderr)

        ##################################################
        # Variables
        ##################################################
        self.snr_db = snr_db = -7
        self.samp_rate = samp_rate = 125000
        self.kind = kind = 2
        self.jsr_db = jsr_db = 10

        ##################################################
        # Blocks
        ##################################################

        self._snr_db_range = qtgui.Range(-20, 10, 1, -7, 200)
        self._snr_db_win = qtgui.RangeWidget(self._snr_db_range, self.set_snr_db, "Desired-signal SNR (dB)", "counter_slider", float, QtCore.Qt.Horizontal)
        self.top_grid_layout.addWidget(self._snr_db_win, 0, 0, 1, 1)
        for r in range(0, 1):
            self.top_grid_layout.setRowStretch(r, 1)
        for c in range(0, 1):
            self.top_grid_layout.setColumnStretch(c, 1)
        # Create the options list
        self._kind_options = [0, 1, 2, 3]
        # Create the labels list
        self._kind_labels = ['Barrage', 'Swept', 'Pulsed', 'Event-triggered']
        # Create the combo box
        self._kind_tool_bar = Qt.QToolBar(self)
        self._kind_tool_bar.addWidget(Qt.QLabel("Interference class" + ": "))
        self._kind_combo_box = Qt.QComboBox()
        self._kind_tool_bar.addWidget(self._kind_combo_box)
        for _label in self._kind_labels: self._kind_combo_box.addItem(_label)
        self._kind_callback = lambda i: Qt.QMetaObject.invokeMethod(self._kind_combo_box, "setCurrentIndex", Qt.Q_ARG("int", self._kind_options.index(i)))
        self._kind_callback(self.kind)
        self._kind_combo_box.currentIndexChanged.connect(
            lambda i: self.set_kind(self._kind_options[i]))
        # Create the radio buttons
        self.top_grid_layout.addWidget(self._kind_tool_bar, 0, 2, 1, 1)
        for r in range(0, 1):
            self.top_grid_layout.setRowStretch(r, 1)
        for c in range(2, 3):
            self.top_grid_layout.setColumnStretch(c, 1)
        self._jsr_db_range = qtgui.Range(0, 20, 1, 10, 200)
        self._jsr_db_win = qtgui.RangeWidget(self._jsr_db_range, self.set_jsr_db, "Interference JSR (dB)", "counter_slider", float, QtCore.Qt.Horizontal)
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
            'Received chirp signal waterfall', #name
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

        self.top_grid_layout.addWidget(self._qtgui_waterfall_sink_x_0_win, 1, 2, 2, 2)
        for r in range(1, 3):
            self.top_grid_layout.setRowStretch(r, 1)
        for c in range(2, 4):
            self.top_grid_layout.setColumnStretch(c, 1)
        self.qtgui_number_sink_0 = qtgui.number_sink(
            gr.sizeof_float,
            0,
            qtgui.NUM_GRAPH_NONE,
            5,
            None # parent
        )
        self.qtgui_number_sink_0.set_update_time(0.25)
        self.qtgui_number_sink_0.set_title('Recovered packet outcomes (0/1)')

        labels = ['True onset (sample)', 'Unmodified: valid packet', 'Blanking: valid packet', 'Spectral: valid packet', 'Combined: valid packet',
            '', '', '', '', '']
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
        self.top_grid_layout.addWidget(self._qtgui_number_sink_0_win, 3, 0, 1, 4)
        for r in range(3, 4):
            self.top_grid_layout.setRowStretch(r, 1)
        for c in range(0, 4):
            self.top_grid_layout.setColumnStretch(c, 1)
        self.qtgui_freq_sink_x_0 = qtgui.freq_sink_c(
            1024, #size
            window.WIN_BLACKMAN_hARRIS, #wintype
            0, #fc
            samp_rate, #bw
            'Received chirp signal spectrum', #name
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



        labels = ['Received IQ', '', '', '', '',
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
        self.top_grid_layout.addWidget(self._qtgui_freq_sink_x_0_win, 1, 0, 2, 2)
        for r in range(1, 3):
            self.top_grid_layout.setRowStretch(r, 1)
        for c in range(0, 2):
            self.top_grid_layout.setColumnStretch(c, 1)
        self.epy_block_0 = epy_block_0.blk(snr_db=snr_db, jsr_db=jsr_db, kind=kind, seed=2026)


        ##################################################
        # Connections
        ##################################################
        self.connect((self.epy_block_0, 0), (self.qtgui_freq_sink_x_0, 0))
        self.connect((self.epy_block_0, 4), (self.qtgui_number_sink_0, 3))
        self.connect((self.epy_block_0, 1), (self.qtgui_number_sink_0, 0))
        self.connect((self.epy_block_0, 3), (self.qtgui_number_sink_0, 2))
        self.connect((self.epy_block_0, 5), (self.qtgui_number_sink_0, 4))
        self.connect((self.epy_block_0, 2), (self.qtgui_number_sink_0, 1))
        self.connect((self.epy_block_0, 0), (self.qtgui_waterfall_sink_x_0, 0))


    def closeEvent(self, event):
        self.settings = Qt.QSettings("GNU Radio", "Lab4_Chirp_Receiver")
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
        self.qtgui_waterfall_sink_x_0.set_frequency_range(0, self.samp_rate)
        self.qtgui_freq_sink_x_0.set_frequency_range(0, self.samp_rate)

    def get_kind(self):
        return self.kind

    def set_kind(self, kind):
        self.kind = kind
        self._kind_callback(self.kind)
        self.epy_block_0.kind = self.kind

    def get_jsr_db(self):
        return self.jsr_db

    def set_jsr_db(self, jsr_db):
        self.jsr_db = jsr_db
        self.epy_block_0.jsr_db = self.jsr_db




def main(top_block_cls=Lab4_Chirp_Receiver, options=None):

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
