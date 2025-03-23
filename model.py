from flask_login import UserMixin

class User(UserMixin):
    def __init__(self, user_dict):
        self.id = user_dict['userID']
        self.firstName = user_dict['firstName']
        self.lastName = user_dict['lastName']
        self.email = user_dict['email']
        self.password = user_dict['password']

