#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2019 Satpy developers
#
# This file is part of satpy.
#
# satpy is free software: you can redistribute it and/or modify it under the
# terms of the GNU General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your option) any later
# version.
#
# satpy is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
# A PARTICULAR PURPOSE.  See the GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License along with
# satpy.  If not, see <http://www.gnu.org/licenses/>.

"""Tests for the Icare MeteoFrance netcdfs reader,
	satpy/readers/geos_netcdficare.py.

python .../site-packages/satpy/tests/reader_tests/test_geos_netcdfcare.py /tmp

'/tmp' is a location for a tempory file written during the test.

EXECUTION TIME :
	36 seconds.
DATE OF CREATION :
	2024 2th october.
LAST VERSIONS :

AUTHOR :
	Meteo France.
"""

import os

import numpy as np

from satpy.readers import load_reader

from satpy.scene import Scene
from satpy import find_files_and_readers
from satpy._config import config_search_paths

from datetime import datetime
from netCDF4 import Dataset
import sys


class TestGeosNetcdfIcareReader() :
	# Test of the geos_netcdficare reader.
	# This reader has been build for the Icare Meteo France netcdfs.

	def assertEqual(self, arg1, arg2) :
		if arg1 != arg2 :
			print(arg1, " not equal to ", arg2)
			exit(1)

	def assertTrue(self, arg) :
		if arg is None :
			print(arg, " not true.")
			exit(1)

	def testStartEndTime(self, expectedStartTime, expectedEndTime) :

		startTime = self.scn.start_time
		startTimeString = startTime.strftime('%Y-%m-%dT%H:%M:%S')
		# 2024-06-28T10:00:40
		self.assertEqual(startTimeString, expectedStartTime)

		endTime = self.scn.end_time
		endTimeString = endTime.strftime('%Y-%m-%dT%H:%M:%S')
		# 2024-06-28T10:12:41
		self.assertEqual(endTimeString, expectedEndTime)

	def testPlateform(
		self, expectedPlatform, expectedSensor,
		expectedAltitude, expectedLongitude) :

		sensor = self.scn.sensor_names
		for isensor in sensor :
			capteur = isensor
		self.assertEqual(capteur, expectedSensor)

		platform = "erreur"
		altitude = -1.
		longitude = 999.

		for data_arr in self.values :
			# values come from the scene.
			# print("data_arr.attrs = ", data_arr.attrs)
			if "platform_name" in data_arr.attrs :
				platform = data_arr.attrs["platform_name"]
			if "orbital_parameters" in data_arr.attrs :
				subAttr = data_arr.attrs["orbital_parameters"]
				if "satellite_actual_altitude" in subAttr :
					altitude = subAttr["satellite_actual_altitude"]
			if "satellite_actual_longitude" in data_arr.attrs :
				longitude = data_arr.attrs["satellite_actual_longitude"]

		longitude = float(int(longitude * 10.)) / 10.
		self.assertEqual(platform, expectedPlatform)
		self.assertEqual(longitude, expectedLongitude)
		self.assertEqual(altitude, expectedAltitude)
		# testPlateform()

	def testResolution(
		self, expectedResolution, expectedNblin, expectedNbpix,
		expectedCfac, expectedLfac, expectedCoff, expectedLoff) :

		xr = self.scn.to_xarray_dataset()
		# print("xr = ", xr)
		matrice = xr["convection"]
		nblin = matrice.shape[1]
		nbpix = matrice.shape[2]
		self.assertEqual(expectedNblin, nblin)
		self.assertEqual(expectedNbpix, nbpix)

		cfac = xr.attrs["cfac"]
		self.assertEqual(expectedCfac, cfac)
		lfac = xr.attrs["lfac"]
		self.assertEqual(expectedLfac, lfac)
		coff = xr.attrs["coff"]
		self.assertEqual(expectedCoff, coff)
		loff = xr.attrs["loff"]
		self.assertEqual(expectedLoff, loff)

		satpyId = xr.attrs["_satpy_id"]
		# DataID(name='convection', resolution=3000.403165817)
		# Cf satpy/dataset/dataid.py.

		resolution = satpyId.get("resolution")
		resolution = float(int(resolution * 10.)) / 10.
		self.assertEqual(expectedResolution, resolution)

	def testVariable(self) :
		# A picture of convection composite will be displayed.
		self.scn.show("convection")
		print("The picture should be pink.")

	def buildNetcdf(self, ncName) :
		"""
		ncName : /tmp/Mmultic3kmNC4_msg03_202406281000.nc
		A dummy icare Meteo France netcdf is built here.	
		"""
		ncfileOut = Dataset(
			ncName, mode="w", clobber=True,
			diskless=False, persist=False, format='NETCDF4')

		ncfileOut.createDimension(u'ny', 3712)
		ncfileOut.createDimension(u'nx', 3712)
		ncfileOut.createDimension(u'numerical_count', 65536)
		ncfileOut.setncattr("time_coverage_start", "2024-06-28T10:00:09Z383")
		ncfileOut.setncattr("time_coverage_end", "2024-06-28T10:12:41Z365")
		ncfileOut.setncattr("Area_of_acquisition", "globe")

		fill_value = -32768
		var = ncfileOut.createVariable(
			"satellite", "c", zlib=True, complevel=4,
			shuffle=True, fletcher32=False, contiguous=False,
			chunksizes=None, endian='native', least_significant_digit=None)

		var.setncattr("id", "msg03")
		var.setncattr("dst", 35786691.)
		var.setncattr("lon", float(0.1))

		var = ncfileOut.createVariable(
			"geos", "c", zlib=True, complevel=4, shuffle=True,
			fletcher32=False, contiguous=False, chunksizes=None,
			endian='native', least_significant_digit=None)
		var.setncattr("longitude_of_projection_origin", 0.)

		var = ncfileOut.createVariable(
			"GeosCoordinateSystem", "c", zlib=True, complevel=4,
			shuffle=True, fletcher32=False, contiguous=False,
			chunksizes=None, endian='native', least_significant_digit=None)
		var.setncattr(
			"GeoTransform",
			"-5570254, 3000.40604, 0, 5570254, 0, -3000.40604")

		var = ncfileOut.createVariable(
			"ImageNavigation", "c", zlib=True, complevel=4,
			shuffle=True, fletcher32=False, contiguous=False,
			chunksizes=None, endian='native', least_significant_digit=None)
		var.setncattr("CFAC", 1.3642337E7)
		var.setncattr("LFAC", 1.3642337E7)
		var.setncattr("COFF", 1857.0)
		var.setncattr("LOFF", 1857.0)

		var = ncfileOut.createVariable(
			"X", 'float32', u'nx', zlib=True, complevel=4,
			shuffle=True, fletcher32=False, contiguous=False,
			chunksizes=None, endian='native', least_significant_digit=None)
		x0 = -5570254.
		dx = 3000.40604
		var[:] = np.array(([(x0 + dx * i) for i in range(3712)]))

		var = ncfileOut.createVariable(
			"Y", 'float32', u'ny', zlib=True, complevel=4,
			shuffle=True, fletcher32=False, contiguous=False,
			chunksizes=None, endian='native', least_significant_digit=None)
		y0 = 5570254.
		dy = -3000.40604
		var[:] = np.array(([(y0 + dy * i) for i in range(3712)]))

		for channel in {"VIS006", "VIS008", "IR_016"} :
			var = ncfileOut.createVariable(
				channel, 'short', ('ny', 'nx'), zlib=True, complevel=4,
				shuffle=True, fletcher32=False, contiguous=False,
				chunksizes=None, endian='native',
				least_significant_digit=None, fill_value=fill_value)
			var[:] = np.array(([[i * 2 for i in range(3712)] for j in range(3712)]))
			# Hundredths of albedo between 0 and 10000.
			var.setncattr("scale_factor", 0.01)
			var.setncattr("add_offset", 0.)
			var.setncattr("bandfactor", 20.76)
			var.setncattr("_CoordinateSystems", "GeosCoordinateSystem")

			var = ncfileOut.createVariable(
				"Albedo_to_Native_count_" + channel, 'short',
				'numerical_count', zlib=True, complevel=4, shuffle=True,
				fletcher32=False, contiguous=False, chunksizes=None,
				endian='native', least_significant_digit=None,
				fill_value=-9999)
			var[:] = np.array(([-9999 for i in range(65536)]))
			# In order to come back to the native datas on 10, 12 or 16 bits.

		for channel in {
			"IR_039", "WV_062", "WV_073", "IR_087", "IR_097",
			"IR_108", "IR_120", "IR_134"} :
			var = ncfileOut.createVariable(
				channel, 'short', ('ny', 'nx'), zlib=True, complevel=4,
				shuffle=True, fletcher32=False, contiguous=False,
				chunksizes=None, endian='native',
				least_significant_digit=None, fill_value=fill_value)
			var[:] = np.array(
				([[-9000 + j * 4 for i in range(3712)] for j in range(3712)]))
			# Hundredths of celcius degrees.
			var.setncattr("scale_factor", 0.01)
			var.setncattr("add_offset", 273.15)
			var.setncattr("nuc", 1600.548)
			var.setncattr("alpha", 0.9963)
			var.setncattr("beta", 2.185)
			var.setncattr("_CoordinateSystems", "GeosCoordinateSystem")

			var = ncfileOut.createVariable(
				"Temp_to_Native_count_" + channel, 'short',
				'numerical_count', zlib=True, complevel=4, shuffle=True,
				fletcher32=False, contiguous=False, chunksizes=None,
				endian='native', least_significant_digit=None,
				fill_value=-9999)
			var[:] = np.array(([-9999 for i in range(65536)]))
			# In order to come back to the native datas on 10, 12 or 16 bits.
		ncfileOut.close
		# buildNetcdf()

	def init(self, netcdfName) :
		"""
		netcdfName : Mmultic3kmNC4_msg03_202406281000.nc
		A scene is built with the reader to be tested, applied to this netcdf.
		"""

		self.yaml_file = 'msg_netcdficare'
		self.reader_configs = config_search_paths(
			os.path.join('readers', self.yaml_file + ".yaml"))
		self.r = load_reader(self.reader_configs)

		loadables = self.r.select_files_from_pathnames([netcdfName])

		# loadables =  ['/tmp//Mmultic3kmNC4_msg03_202406281000.nc']

		self.assertEqual(len(loadables), 1)

		self.r.create_filehandlers(loadables)

		self.assertTrue(self.r.file_handlers)
		# {'MSG3km': [<NETCDF_ICARE: '/tmp/Mmultic3kmNC4_msg03_202406281000.nc'>]}

		myfiles = find_files_and_readers(
			base_dir="/tmp/", start_time=datetime(2024, 6, 28, 10, 0),
			end_time=datetime(2024, 6, 28, 10, 0), reader=self.yaml_file)
		print("Found myfiles = ", myfiles)
		# {'msg_netcdficare': ['/tmp/Mmultic3kmNC4_msg03_202406281000.nc']}

		self.scn = Scene(filenames=myfiles, reader=self.yaml_file)

		print(self.scn.available_dataset_names())
		# ['IR_016', 'IR_039', 'IR_087', 'IR_097', 'IR_108', 'IR_120',
		# 'IR_134', 'VIS006', 'VIS008', 'WV_062', 'WV_073']

		print(self.scn.available_composite_names())
		""" Static compositor  {'_satpy_id': DataID(name='_night_background'),
		'standard_name': 'night_background',
		'prerequisites': [],
		'optional_prerequisites': []}
		Static compositor  {'_satpy_id': DataID(name='_night_background_hires'),
		'standard_name': 'night_background', 'prerequisites': [],
		'optional_prerequisites': []}
		['airmass', 'ash', 'cloud_phase_distinction', 'cloud_phase_distinction_raw',
		'cloudtop', 'cloudtop_daytime', 'colorized_ir_clouds', 'convection'...
		"""

		self.scn.load(['convection'])
		self.values = self.scn.values()
		# init()

	# class TestGeosNetcdfIcareReader.


if len(sys.argv) > 1 :
	filepath = sys.argv[1]
else :
	filepath = "/tmp"

instanceTest = TestGeosNetcdfIcareReader()

netcdfName = filepath + "/Mmultic3kmNC4_msg03_202406281000.nc"

# Building of a dummy netcdf, used to test the reader.
instanceTest.buildNetcdf(netcdfName)

# The reader will decode the dummy file netcdfName. A scene is built.
instanceTest.init(netcdfName)

# We check that the parameters written in the dummy netcdf can be read.
instanceTest.testStartEndTime("2024-06-28T10:00:09", "2024-06-28T10:12:41")

actualAltitude = 35786691 + 6378169		# 42164860.0
actualLongitude = 0.1
instanceTest.testPlateform(
	"Meteosat-10", "seviri", actualAltitude, actualLongitude)

cfac = 1.3642337E7
lfac = 1.3642337E7
coff = 1857.0
loff = 1857.0
instanceTest.testResolution(3000.4, 3712, 3712, cfac, lfac, coff, loff)

instanceTest.testVariable()

print("The test result is correct.")
