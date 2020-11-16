#!/usr/bin/env python
# coding: utf-8

import telebot
token = '1272224178:AAFmjCgTbCZtz2FmrSk-qeQRkf4-On0MOx8'
import sqlite3

bot = telebot.TeleBot(token)


@bot.message_handler(commands=['start'])
def start_message(message):
    bot.send_message(message.chat.id, 'Привет, ты написал мне /start')


def check_address(message):
    if message.text.count(",")  == 2:
        return True
    return False

def check_show_places(message):
    if message.text  == 'покажи места':
        return True
    return False

def delete_places(message):
    if message.text  == 'удали данные':
        return True
    return False


@bot.message_handler(commands=['add'])
def handle_address_com(message):
    bot.send_message(message.chat.id, 'Введите название города, улицы и дома через запятую в \
            формате: Нью-Йорк, 3е Авеню, 2')


@bot.message_handler(func=check_address)
def handle_address(message):
    data_list = message.text.split(',')
    city, street, house =  data_list[0], data_list[1], data_list[2]
    bot.send_message(message.chat.id, 'Место по адресу: г.{}, ул.{}, д.{} сохранено'.format(city, street, house))
    with sqlite3.connect("mydatabase.db", check_same_thread=False) as conn:
        cursor = conn.cursor()
        cursor.execute("INSERT INTO places VALUES (?,?,?)", data_list)
        conn.commit()


@bot.message_handler(commands=['list'])
@bot.message_handler(func=check_show_places)
def handle_list_of_places(message):
    with sqlite3.connect("mydatabase.db", check_same_thread=False) as conn:
        cursor = conn.cursor()
        sql = "SELECT * FROM places LIMIT 10"
        cursor.execute(sql)
        for now in cursor.fetchall():
            bot.send_message(message.chat.id, 'г.{}, ул.{}, д.{}'.format(now[0], now[1], now[2]))


@bot.message_handler(commands=['delete'])
@bot.message_handler(func=delete_places)
def handle_list_of_places(message):
    with sqlite3.connect("mydatabase.db", check_same_thread=False) as conn:
        cursor.execute("TRUNCATE TABLE places")
        conn.commit()


@bot.message_handler()
def handle_message(message):
    if message.text.lower() == 'привет':
        bot.send_message(message.chat.id, 'Привет, мой повелитель')
    elif message.text.lower() == 'пока':
        bot.send_message(message.chat.id, 'Прощай, мой повелитель')
    elif message.text.lower() == 'добавить место':
        bot.send_message(message.chat.id, 'Введите название города, улицы и дома через запятую в \
            формате Нью-Йорк, 3е Авеню, 2')
    else:
        bot.send_message(message.chat.id, text='Ну и? Я тебя не понимаю, научись писать нормально!')


bot.polling()
