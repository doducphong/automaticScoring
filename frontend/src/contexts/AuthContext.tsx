'use client';
import React, { createContext, useContext, useState, useEffect, ReactNode } from 'react';
import Cookies from 'js-cookie';
import { useRouter } from 'next/navigation';
import axios from 'axios';

axios.defaults.withCredentials = true;

type User = {
  id: string;
  email: string;
  name: string;
  is_active: boolean;
};

type AuthContextType = {
  user: User | null;
  loading: boolean;
  login: (email: string, password: string) => Promise<void>;
  register: (userData: RegisterData) => Promise<void>;
  logout: () => void;
};

type RegisterData = {
  email: string;
  password: string;
  name: string;
};

const AuthContext = createContext<AuthContextType | undefined>(undefined);

export const AuthProvider = ({ children }: { children: ReactNode }) => {
  const [user, setUser] = useState<User | null>(null);
  const [loading, setLoading] = useState(true);
  const router = useRouter();

  useEffect(() => {
    const loadUserFromToken = async () => {
      const token = Cookies.get('token');
      if (token) {
        try {
          const response = await axios.get('/api/v1/teachers/me');
          setUser(response.data);
        } catch (error) {
          console.error('Failed to get user info', error);
          Cookies.remove('token');
        }
      }
      setLoading(false);
    };

    loadUserFromToken();
  }, []);

  const login = async (email: string, password: string) => {
    setLoading(true);
    try {
      const formData = new FormData();
      formData.append('username', email);
      formData.append('password', password);

      const response = await axios.post('/api/v1/auth/login', formData);
      const { access_token, token_type } = response.data;
      
      // Save token to cookies
      Cookies.set('token', access_token, { expires: 1 }); // 1 day expiry

      // Get user info
      const userResponse = await axios.get('/api/v1/teachers/me');
      
      setUser(userResponse.data);
      router.push('/dashboard');
    } catch (error) {
      console.error('Login failed:', error);
      throw new Error('Login failed. Please check your credentials.');
    } finally {
      setLoading(false);
    }
  };

  const register = async (userData: RegisterData) => {
    setLoading(true);
    try {
      await axios.post('/api/v1/auth/register', userData);
      // After successful registration, login the user
      await login(userData.email, userData.password);
    } catch (error) {
      console.error('Registration failed:', error);
      throw new Error('Registration failed. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  const logout = () => {
    Cookies.remove('token');
    setUser(null);
    router.push('/login');
  };

  return (
    <AuthContext.Provider value={{ user, loading, login, register, logout }}>
      {children}
    </AuthContext.Provider>
  );
};

export const useAuth = () => {
  const context = useContext(AuthContext);
  if (context === undefined) {
    throw new Error('useAuth must be used within an AuthProvider');
  }
  return context;
};
