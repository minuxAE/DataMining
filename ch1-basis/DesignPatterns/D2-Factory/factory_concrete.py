from abc import ABCMeta

class Section(metaclass=ABCMeta):
    @classmethod
    def describe(self):
        pass

class PersonalSection(Section):
    def describe(self):
        print('Personal Section')

class AlbumSection(Section):
    def describe(self):
        print('Album Section')

class PatentSection(Section):
    def describe(self):
        print('Patent Section')

class PublicationSection(Section):
    def describe(self):
        print('Publication Section')

class Profile(metaclass=ABCMeta):
    def __init__(self) -> None:
        self.sections = []
        self.createProfile()

    @classmethod
    def createProfile(self):
        pass

    def getSections(self):
        return self.sections
    
    def addSections(self, section):
        return self.sections.append(section)

class linkedin(Profile):
    def createProfile(self):
        self.addSections(PersonalSection())
        self.addSections(PatentSection())
        self.addSections(PublicationSection())

class facebook(Profile):
    def createProfile(self):
        self.addSections(PersonalSection())
        self.addSections(AlbumSection())


if __name__ == '__main__':
    profile_type = input('Which Profile you would like to create? Linkedin / Facebook\n')
    profile = eval(profile_type.lower())()
    print('Creating Profile..', type(profile).__name__)
    print('Profile has sections --', profile.getSections())