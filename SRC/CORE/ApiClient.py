from abc import ABC, abstractmethod
import requests


class ApiClient(ABC):
	@abstractmethod
	def get_historical_klines(self, start_time, end_time):
		pass

	@abstractmethod
	def get_end_point(self):
		pass

	@abstractmethod
	def minutes_of_new_data(self, data):
		pass