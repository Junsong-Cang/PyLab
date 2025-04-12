# Examples of class
class Car():
    '''Define a car'''
    def __init__(self, make, model, year='2004'):#可设置默认参数
        self.make = make
        self.model = model
        self.year = year
        self.odometer_reading = 0 # 指定默认值，无需加入init形参
    def Get_Info(self):
        '''Tell me about the car'''
        long_name = str(self.year) + ' ' + self.make + ' ' + self.model
        return long_name
    def update_odometer(self, mileage):
        self.odometer_reading = mileage
    def Fill_Gas_Tank(self, q):
        s = 'Filled gas tank with '+q
        return s
'''
a = Car('Bugatti', model='TOURBILLON', year = '2024')
a.odometer_reading = 100
print(a.odometer_reading)
a.update_odometer(1000)
print(a.odometer_reading)
'''
class ElectricCar(Car):
    def __init__(self, make, model, year, battery_size):
        super().__init__(make, model, year)
        self.battery_size = battery_size
    def show_info(self):
        s1 = self.Get_Info()
        s2 = ', battery = ' + self.battery_size
        return s1 + s2

a = ElectricCar('Farrari', 'Stradale', '2024', '1000')
print(a.show_info())
print(a.Fill_Gas_Tank('U235'))
