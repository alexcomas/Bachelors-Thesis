from turtle import color
import colorama
from scipy.io import arff
import glob
import csv
from datetime import datetime, date, time, timedelta
import sys
import os
import pandas as pd

attack_table = pd.DataFrame(columns=['Attacker','Victim','Attack Name','Date','Attack Start Time','Attack Finish Time'])

def appendAttack(df, attackers, victims, attack_name, indate, start_time, finish_time):
    return df.append({'Attacker': attackers, 'Victim': victims, 'Attack Name': attack_name, 
        'Date': datetime.strptime(indate, "%a-%d-%m-%Y").date() , 
        'Attack Start Time': datetime.strptime(start_time, "%H:%M").time(), 
        'Attack Finish Time': datetime.strptime(finish_time, "%H:%M").time() }, ignore_index=True)
        
attack_table = appendAttack(attack_table, ['18.221.219.4'], ['172.31.69.25'], 'FTP-BruteForce', 'Wed-14-02-2018', '10:32', '12:09')
attack_table = appendAttack(attack_table, ['13.58.98.64'], ['172.31.69.25'], 'SSH-BruteForce', 'Wed-14-02-2018', '14:01', '15:31')
attack_table = appendAttack(attack_table, ['18.219.211.138'], ['172.31.69.25'], 'DoS-GoldenEye', 'Thu-15-02-2018', '9:26', '10:09')
attack_table = appendAttack(attack_table, ['18.217.165.70'], ['172.31.69.25'], 'DoS-Slowloris', 'Thu-15-02-2018', '10:59', '11:40')
attack_table = appendAttack(attack_table, ['13.59.126.31'], ['172.31.69.25'], 'DoS-SlowHTTPTest', 'Fri-16-02-2018', '10:12', '11:08')
attack_table = appendAttack(attack_table, ['18.219.193.20'], ['172.31.69.25'], 'DoS-Hulk', 'Fri-16-02-2018', '13:45', '14:19')
attack_table = appendAttack(attack_table, 
                                            [
                                                '18.218.115.60', 
                                                '18.219.9.1', 
                                                '18.219.32.43', 
                                                '18.218.55.126', 
                                                '52.14.136.135', 
                                                '18.219.5.43', 
                                                '18.216.200.189', 
                                                '18.218.229.235', 
                                                '18.218.11.51',
                                                '18.216.24.42'], 
                                            ['172.31.69.25'], 'DDoS attacks-LOIC-HTTP', 'Tue-20-02-2018', '10:12', '11:17') 
attack_table = appendAttack(attack_table, 
                                            [   '18.218.115.60', 
                                                '18.219.9.1', 
                                                '18.219.32.43', 
                                                '18.218.55.126', 
                                                '52.14.136.135', 
                                                '18.219.5.43', 
                                                '18.216.200.189', 
                                                '18.218.229.235', 
                                                '18.218.11.51',
                                                '18.216.24.42'], 
                                            ['172.31.69.25'], 'DDoS-LOIC-UDP', 'Tue-20-02-2018', '13:13', '13:32')
attack_table = appendAttack(attack_table, 
                                            [   '18.218.115.60', 
                                                '18.219.9.1', 
                                                '18.219.32.43', 
                                                '18.218.55.126', 
                                                '52.14.136.135', 
                                                '18.219.5.43', 
                                                '18.216.200.189', 
                                                '18.218.229.235', 
                                                '18.218.11.51',
                                                '18.216.24.42'], 
                                            ['172.31.69.28'], 'DDoS-LOIC-UDP', 'Wed-21-02-2018', '10:09', '10:43')
attack_table = appendAttack(attack_table, 
                                            [   '18.218.115.60', 
                                                '18.219.9.1', 
                                                '18.219.32.43', 
                                                '18.218.55.126', 
                                                '52.14.136.135', 
                                                '18.219.5.43', 
                                                '18.216.200.189', 
                                                '18.218.229.235', 
                                                '18.218.11.51',
                                                '18.216.24.42'], 
                                            ['172.31.69.28'], 'DDOS-HOIC', 'Wed-21-02-2018', '14:05', '15:05')
attack_table = appendAttack(attack_table, ['18.218.115.60'], ['172.31.69.28'], 'Brute Force -Web', 'Thu-22-02-2018', '10:17', '11:24')
attack_table = appendAttack(attack_table, ['18.218.115.60'], ['172.31.69.28'], 'Brute Force -XSS', 'Thu-22-02-2018', '13:50', '14:29')
attack_table = appendAttack(attack_table, ['18.218.115.60'], ['172.31.69.28'], 'SQL Injection', 'Thu-22-02-2018', '16:15', '16:29')
attack_table = appendAttack(attack_table, ['18.218.115.60'], ['172.31.69.28'], 'Brute Force -Web', 'Fri-23-02-2018', '10:03', '11:03')
attack_table = appendAttack(attack_table, ['18.218.115.60'], ['172.31.69.28'], 'Brute Force -XSS', 'Fri-23-02-2018', '13:00', '14:10')
attack_table = appendAttack(attack_table, ['18.218.115.60'], ['172.31.69.28'], 'SQL Injection', 'Fri-23-02-2018', '15:05', '15:18')
attack_table = appendAttack(attack_table, ['13.58.225.34'], ['172.31.69.24'], 'Infiltration', 'Wed-28-02-2018', '10:50', '12:05')
attack_table = appendAttack(attack_table, ['13.58.225.34'], ['172.31.69.24'], 'Infiltration', 'Wed-28-02-2018', '13:42', '14:40')
attack_table = appendAttack(attack_table, ['13.58.225.34'], ['172.31.69.13'], 'Infiltration', 'Thu-01-03-2018', '09:57', '10:55')
attack_table = appendAttack(attack_table, ['13.58.225.34'], ['172.31.69.13'], 'Infiltration', 'Thu-01-03-2018', '14:00', '15:37')
attack_table = appendAttack(attack_table,  
                                                            [   '172.31.69.23',
                                                                '172.31.69.17',
                                                                '172.31.69.14',
                                                                '172.31.69.12',
                                                                '172.31.69.10',
                                                                '172.31.69.8',
                                                                '172.31.69.6',
                                                                '172.31.69.26',
                                                                '172.31.69.29',
                                                                '172.31.69.30'], ['18.219.211.138'],
                                                            'Bot', 'Fri-02-03-2018', '10:11', '11:34')
attack_table = appendAttack(attack_table, 
                                                            [   '172.31.69.23',
                                                                '172.31.69.17',
                                                                '172.31.69.14',
                                                                '172.31.69.12',
                                                                '172.31.69.10',
                                                                '172.31.69.8',
                                                                '172.31.69.6',
                                                                '172.31.69.26',
                                                                '172.31.69.29',
                                                                '172.31.69.30'], ['18.219.211.138'], 
                                                            'Bot', 'Fri-02-03-2018', '14:24', '15:55')
datetimeformat = "%d/%m/%Y %I:%M:%S %p"

def getAttack(intimestamp: str, attacker, victim, tolerance = timedelta(minutes=2)):
    indatetime = datetime.strptime(intimestamp, datetimeformat)
    indatetime = indatetime - timedelta(hours=5) # La taula dona el temps en local mentre els PCAP es registren en un altre zona horaria amb 5 hores de diferencia
    if len(attack_table[attack_table['Date'] == indatetime.date()]) == 0:
        return 'DISCARD'
    possible_attacks = attack_table[(attack_table['Date'] == indatetime.date()) 
                                    & (attack_table['Attack Start Time'] <= (indatetime+tolerance).time())
                                    & (attack_table['Attack Finish Time'] >= (indatetime-tolerance).time())
                                    & ([attacker in l or 'any' in l for l in attack_table['Attacker']])
                                    & ([victim in l for l in attack_table['Victim']])]
    if len(possible_attacks) == 0:
        return 'BENIGN'
    return possible_attacks['Attack Name'].iloc[0]