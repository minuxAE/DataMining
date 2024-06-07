from abc import abstractclassmethod, ABCMeta

class Trip(metaclass=ABCMeta):
    @abstractclassmethod
    def setTransport(self):
        pass

    @abstractclassmethod
    def day1(self):
        pass

    @abstractclassmethod
    def day2(self):
        pass

    @abstractclassmethod
    def day3(self):
        pass

    @abstractclassmethod
    def returnHome(self):
        pass

    def itinerary(self):
        self.setTransport()
        self.day1()
        self.day2()
        self.day3()
        self.returnHome()


class VeniceTrip(Trip):
    def setTransport(self):
        print('Take a boat and find your way in the Grand  Canal')

    def day1(self):
        print('Visit St Mark Basilica')

    def day2(self):
        print('Appreciate Doge Palace')

    def day3(self):
        print('Enjoy Food near the Rialtio Bridge')

    def returnHome(self):
        print('Get souvenirs for Friends and get Back')


class MaldivesTrip(Trip):
    def setTransport(self):
        print('On foot, on any island, WoW!')

    def day1(self):
        print('Enjoy the marine life of Banana Reef')

    def day2(self):
        print('Go For Water Sports and Snorkelling')

    def day3(self):
        print('Relax on the Beach and Enjoy the sun')

    def returnHome(self):
        print('Dont feel like leaving the beach...')

class TravelAgency:
    def arrange_trip(self):
        choice = input('What kind of place you would like to go: historical or beach:\n')
        if choice == 'historical':
            self.trip = VeniceTrip()
            self.trip.itinerary()
        if choice == 'beach':
            self.trip = MaldivesTrip()
            self.trip.itinerary()

TravelAgency().arrange_trip()

        
