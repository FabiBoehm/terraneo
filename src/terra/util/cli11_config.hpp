
#pragma once

#include <sstream>

#include "cli11_wrapper.hpp"

class ConfigGroupedNoDescriptions : public CLI::ConfigBase
{
  public:
    std::string
        to_config( const CLI::App* app, bool default_also, bool write_description, std::string prefix ) const override
    {
        std::stringstream out;
        std::string       last_group;

        // Header
        if ( !app->get_description().empty() )
        {
            out << "# " << app->get_description() << "\n\n";
        }

        for ( const CLI::Option* opt : app->get_options( {} ) )
        {
            // Skip help/config-file-triggering options
            if ( opt == app->get_help_ptr() || opt == app->get_config_ptr() )
                continue;

            const bool has_value = opt->count() > 0;
            if ( !default_also && !has_value )
                continue;

            const std::string& group = opt->get_group();

            // Skip empty-group options, which will be internally set to CLI11 default group "OPTIONS".
            if ( group == "OPTIONS" || group == "Options" )
                continue;

            if ( group != last_group )
            {
                if ( !last_group.empty() )
                    out << "\n";
                out << "# ---- " << group << " ----\n";
                last_group = group;
            }

            // get_lnames() returns long names without the leading "--"; fall back to get_name() if cli11 version doesn't expose get_lnames() directly.
            const auto&       lnames = opt->get_lnames();
            const std::string key    = lnames.empty() ? opt->get_name() : lnames.front();

            // Write option descriptions
            if ( write_description && !opt->get_description().empty() )
            {
                out << "# " << opt->get_description() << "\n";
            }

            out << prefix << key << " = ";

            const CLI::results_t& vals = opt->results();
            if ( vals.empty() )
            {
                out << opt->get_default_str();
            }
            else
            {
                for ( size_t i = 0; i < vals.size(); ++i )
                {
                    if ( i )
                        out << " ";
                    out << vals[i];
                }
            }
            out << "\n";
        }

        return out.str();
    }
};
