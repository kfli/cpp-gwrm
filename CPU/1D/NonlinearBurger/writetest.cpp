#include <iostream>
#include <fstream>
#include <string>

const int WIDTH = 2;
const int HEIGHT = 3;

// https://en.cppreference.com/w/cpp/language/type_alias
using array_type = unsigned char[HEIGHT][WIDTH] ;

bool writeImage( const array_type& array, const std::string& file_name );

int main() {

    const std::string file_name = "array.txt";

    const array_type array = { {1,2}, {2,3}, {5,1} };

    if( writeImage( array, file_name ) ) {

       #ifndef NDEBUG // if the program is being debugged

            // print a success message and dump the contents of the file on stdout
            std::cout << "write successful.\ncontents of file '" << file_name
                      << "' :\n-------------\n" ;

            // dump the contents of the file (dump the entire stream buffer)
            // https://en.cppreference.com/w/cpp/io/basic_ios/rdbuf
            std::cout << std::ifstream(file_name).rdbuf() << "-------------\n" ;

       #endif
    }

    else {

        std::cerr << "write failed\n" ;
        return 1 ; // return non-zero from main to indicate exit with failure
    }
}

bool writeImage( const array_type& array, const std::string& file_name ) {

    // if the file was opened successfully for output
    if( std::ofstream file{file_name} ) {

        // http://www.stroustrup.com/C++11FAQ.html#for
        for( const auto& row : array ) { // for each row in the file

            // write each value in the row (as integer)
            for( unsigned int value : row ) {
				file << value << " ";
			}
            file << "\n" ; // and put a new line at the end of the row
        }

        return file.good() ; // write was successful if the file is still in a good state
    }

    else return false ; // failed to open file for output
}