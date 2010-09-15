/*
 * auto-table.h
 *
 *  Created on: Sep 15, 2010
 *      Author: baj
 */

#ifndef AUTO_TABLE_H_
#define AUTO_TABLE_H_

#include <vector>
#include <map>

template<class KeyType, class DataType>
class AutoTable {
public:
	void save(const char *file_name) const {
		std::ofstream fout(file_name);

		if (fout.good()) {
			for (typename std::map<KeyType, DataType>::const_iterator it = table_.begin(); it != table_.end(); ++it) {
				fout << std::setprecision(13) << it->first << " " << it->second << std::endl;
			}
		}

		fout.close();
	}

	void load(const char *file_name) {
		std::ifstream fin(file_name);

		if (fin.good()) {
			KeyType key;
			DataType data;

			while (!fin.eof()) {
				fin >> key >> data;
				table_[key] = data;
			}
		}

		fin.close();
	}

	std::map<KeyType, DataType> & table() { return table_; }

private:
	std::map<KeyType, DataType> table_;
};


#endif /* AUTO_TABLE_H_ */
