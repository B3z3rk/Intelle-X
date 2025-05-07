DROP DATABASE IF EXISTS intellexDB;
CREATE DATABASE IF NOT EXISTS intellexDB;

USE intellexDB;

CREATE TABLE user(
    userID INT AUTO_INCREMENT PRIMARY KEY,
    firstName VARCHAR(100) NOT NULL,
    lastName VARCHAR(100) NOT NULL,
    email VARCHAR(100) NOT NULL UNIQUE,
    verified TINYINT(1) DEFAULT 0,
    password VARCHAR(200) NOT NULL
);

CREATE TABLE history (
    hid INT AUTO_INCREMENT PRIMARY KEY,
    userID INT NOT NULL,
    query TEXT NOT NULL,
    docIndex INT NOT NULL,
    filename VARCHAR(255) NOT NULL,
    matching_sections TEXT NOT NULL,
    score FLOAT NOT NULL,
    paraphrased_response TEXT NOT NULL,
    FOREIGN KEY (userID) REFERENCES user(userID) ON DELETE CASCADE
);
